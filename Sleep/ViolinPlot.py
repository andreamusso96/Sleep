from typing import Dict

import xarray as xr
from typing import List
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import mobile_traffic as mt
import insee
import numpy as np
from tqdm import tqdm
from datetime import date, datetime, timedelta, time
from plotly.subplots import make_subplots

from aggregate_geo import get_matching_netmob_tile_to_insee_tile, aggregate_netmob_tile_level_variables_to_insee_tile_level

data_path = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/BaseData'
figure_folder_path = '/Users/andrea/Desktop/PhD/Presentations/HTMLPresentations/reveal.js/assets/2023-10-06-Sleep/'


def load_data():
    data = xr.open_dataset(f'{data_path}/app_consumption_by_location_and_time_of_day_insee_tile.nc')
    data = data.to_array().squeeze()
    admin_data = pd.read_csv(f'{data_path}/admin_data_insee_tile.csv', index_col=0)
    income = (admin_data['Ind_snv'] / admin_data['Ind']).to_frame('income')
    log_income = np.log2(income['income']).to_frame('log_income')
    return data, log_income


def preprocess_data(data: xr.DataArray, log_income: pd.DataFrame, quantiles: List[float]):
    log_income_quantiles = np.quantile(log_income['log_income'], q=quantiles)
    bins = [-np.inf] + log_income_quantiles.tolist() + [np.inf]
    categories = pd.cut(log_income['log_income'], bins=bins, labels=[f'q{k}' for k in range(len(bins) - 1)])
    map_insee_tile_to_category = categories.to_dict()
    data = data.assign_coords(insee_tile=[map_insee_tile_to_category[tile] for tile in data.insee_tile.values])
    data = data.groupby('insee_tile').mean()
    return data


def distribution_plot(data: xr.DataArray, services: np.ndarray, split_point: int):
    data = data.mean(dim='insee_tile')
    times_from_reference_time = sort_times(times=data.time.values)
    base_elements = np.arange(len(data.time.values))
    map_time_to_element = {t: base_elements[k] for k, t in enumerate(times_from_reference_time)}
    map_element_to_time = {v: k for k, v in map_time_to_element.items()}

    fig = make_subplots(rows=1, cols=2)

    services_low = services[:split_point][::-1]
    traces_low = distribution_plot_column(data=data, services=services_low, map_time_to_element=map_time_to_element)
    fig.add_traces(traces_low, rows=1, cols=1)

    services_high = services[-split_point:]
    traces_high = distribution_plot_column(data=data, services=services_high, map_time_to_element=map_time_to_element)
    fig.add_traces(traces_high, rows=1, cols=2)

    fig.update_layout(height=80 * split_point, font=dict(size=20), template='plotly_white', title='App usage distribution over time')
    fig.update_xaxes(ticktext=[map_element_to_time[e] for e in base_elements if e % 4 == 0], tickvals=[b for b in base_elements if b % 4 == 0])
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/app_usage_time_distribution_plot.html')


def distribution_plot_column(data: xr.DataArray, services: List[str], map_time_to_element: Dict[str, int]):
    traces = []
    for service in services:
        data_service = data.sel(service=service).to_pandas()
        probabilities = data_service / data_service.sum()
        elements = np.array([map_time_to_element[t] for t in data_service.index])
        samples = sample_from_distribution(elements=elements, probabilities=probabilities, n_samples=1000)
        trace = go.Violin(x=samples, name=service, legendgroup=service, side='positive', box_visible=True, meanline_visible=True, orientation='h', width=3)
        traces.append(trace)
    return traces


def make_violin_plot(data: xr.DataArray, services: List[str]):
    quantiles = np.unique(data.insee_tile.values)
    quantiles = [quantiles[0], quantiles[-1]]
    map_quantiles_to_side = {quantiles[0]: 'negative', quantiles[-1]: 'positive'}
    times_from_reference_time = sort_times(times=data.time.values)
    base_elements = np.arange(len(data.time.values))
    map_time_to_element = {t: base_elements[k] for k, t in enumerate(times_from_reference_time)}
    map_element_to_time = {v: k for k, v in map_time_to_element.items()}
    map_quantile_to_color = {q: px.colors.qualitative.Plotly[k] for k, q in enumerate(quantiles)}

    n_plots_per_column = 5
    n_rows = int(np.ceil(len(services) / n_plots_per_column))
    fig = make_subplots(rows=n_rows, cols=1)

    for q in quantiles:
        for k in range(0, len(services), n_plots_per_column):
            services_ = services[k:k + n_plots_per_column]
            trace_violin = violin_plot_trace(data=data, quantile=q, services=services_, map_time_to_element=map_time_to_element, color=map_quantile_to_color[q], side=map_quantiles_to_side[q])
            fig.add_trace(trace_violin, row=k // n_plots_per_column + 1, col=1)

    fig.update_layout(font=dict(size=20), height=400 * n_rows, template='plotly_white', title='App usage differences between low and high income areas')
    fig.update_yaxes(ticktext=[map_element_to_time[e] for e in base_elements if e % 4 == 0], tickvals=[b for b in base_elements if b % 4 == 0])
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/app_usage_low_vs_high_income_violin_plot.html')


def violin_plot_trace(data: xr.DataArray, quantile: str, services: List[str], map_time_to_element: Dict[str, int], color: str, side: str):
    x = []
    y = []
    for service in services:
        data_service_q = data.sel(service=service, insee_tile=quantile).to_pandas()
        probabilities = data_service_q / data_service_q.sum()
        elements = np.array([map_time_to_element[t] for t in data_service_q.index])
        samples = sample_from_distribution(elements=elements, probabilities=probabilities, n_samples=1000)
        x += [service] * len(samples)
        y += list(samples)

    trace_violin = go.Violin(x=x, y=y, name=quantile, legendgroup=quantile, side=side, box_visible=True, meanline_visible=True, line_color=color)
    return trace_violin


def sample_from_distribution(elements: np.ndarray, probabilities, n_samples: int) -> np.ndarray:
    return np.random.choice(elements, size=n_samples, replace=True, p=probabilities)


def sort_times(times):
    times_ = [time.fromisoformat(t_str) for t_str in times]
    times_.sort(key=lambda t: _time_from_reference_time(t=t, reference_time=time(21, 0, 0)))
    times_ = [t.isoformat() for t in times_]
    return times_


def _time_from_reference_time(t, reference_time):
    auxiliary_date = date(2020, 1, 1)
    if t < reference_time:
        datetime.combine(auxiliary_date, t)
        return datetime.combine(auxiliary_date, t) + timedelta(days=1) - datetime.combine(auxiliary_date, reference_time)
    else:
        return datetime.combine(auxiliary_date, t) - datetime.combine(auxiliary_date, reference_time)


def aggregate_data() -> xr.DataArray:
    data = xr.open_dataset(f'{data_path}/app_consumption_by_location_and_time_of_day_netmob_tile.nc')
    data = data.to_array().squeeze()
    times = data.time.values
    services = data.service.values

    data_times = []
    for time in tqdm(times):
        data_time = data.sel(time=time)
        data_time = data_time.to_pandas()
        vars_to_aggregate = list(data_time.columns)
        data_time_agg = aggregate_netmob_tile_level_variables_to_insee_tile_level(data=data_time, city_netmob_tile=mt.City.PARIS, variables=vars_to_aggregate, aggregation_fct='weighted_sum')
        data_time_agg.set_index('insee_tile', inplace=True)
        data_time_agg = data_time_agg.loc[data_time_agg['covered_area'] > 0.8].copy()
        data_time_agg.drop(columns=['covered_area'], inplace=True)
        data_times.append(data_time_agg)

    insee_tiles = data_times[0].index
    data_aggregated = np.stack([d.values for d in data_times], axis=-1)
    dims = ['insee_tile', 'service', 'time']
    coords = [insee_tiles, services, times]
    data_aggregated = xr.DataArray(data_aggregated, dims=dims, coords=coords)
    return data_aggregated


def get_ranked_services(selected: bool = True):
    import mobile_traffic as mt
    s = mt.Service
    selected_services = [s.PLAYSTATION, s.SNAPCHAT, s.FORTNITE, s.EA_GAMES, s.WHATSAPP, s.PERISCOPE, s.WEB_STREAMING, s.WEB_ADULT, s.ORANGE_TV, s.WEB_E_COMMERCE, s.TELEGRAM, s.FACEBOOK_MESSENGER, s.FACEBOOK_LIVE, s.WEB_GAMES, s.FACEBOOK, s.CLASH_OF_CLANS, s.PINTEREST, s.WEB_FINANCE, s.TWITTER,
                         s.YOUTUBE, s.TWITCH, s.NETFLIX, s.DAILYMOTION, s.SKYPE, s.MOLOTOV, s.WEB_CLOTHES, s.GOOGLE_DOCS, s.INSTAGRAM, s.DEEZER, s.GOOGLE_DRIVE, s.WEB_FOOD, s.SOUNDCLOUD, s.TOR, s.POKEMON_GO, s.SPOTIFY, s.GOOGLE_MAIL, s.GOOGLE_MEET, s.TEAMVIEWER, s.WIKIPEDIA, s.YAHOO,
                         s.MICROSOFT_MAIL, s.MICROSOFT_OFFICE, s.LINKEDIN]
    selected_services = [s.value for s in selected_services]

    services = pd.read_csv(f'{data_path}/app_rca_ranking_by_low_income.csv', index_col=0)
    if selected:
        services = services.loc[selected_services]

    split_point = len(services.loc[services['low'] > 1])
    service_names = services.index
    return service_names, split_point


if __name__ == '__main__':
    d, i = load_data()
    d = preprocess_data(data=d, log_income=i, quantiles=[0.3, 0.7])
    ser, sp = get_ranked_services(selected=True)
    distribution_plot(data=d, services=ser, split_point=sp)
    make_violin_plot(data=d, services=ser)
