import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as subplots
import plotting

data_folder_path = '/Users/anmusso/Desktop/PhD/Projects/Current/NetMob/NetMobData/data/FigureData/Sleep'
figure_folder_path = '/Users/anmusso/Desktop/PhD/Projects/Current/NetMob/Figures/Sleep'


def paris_map_plots():
    file_name = 'map_figure.geojson'
    data = gpd.read_file(f'{data_folder_path}/{file_name}')

    plt.rcParams['font.size'] = 28
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'medium'
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw={"width_ratios": [1, 1]})
    axes = axes.flatten()
    columns = ['bed_time_index', 'noise_estimate', 'income', 'age']
    col_name = ['Night Screen Index ', 'Noise', 'Median Income', 'Average Age']
    vmin, vmax = -3, 3
    cmap = 'plasma'

    for i in range(len(columns)):
        plotting.map.simple_normalized_map(ax=axes[i], data=data, column=columns[i], cmap=cmap, legend=False, vmin=vmin, vmax=vmax, outlier_threshold=.995)
        axes[i].set_title(col_name[i])

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([]) # this adds in the colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.1, label='Standardized value')
    plt.show()
    fig.savefig(f'{figure_folder_path}/paris_map.png', bbox_inches='tight')


def save_plotly_fig(fig, file_name):
    for i in range(3):
        pio.write_image(fig, f'{figure_folder_path}/{file_name}.png')


def regression_plots():
    file_name = 'regression_figure.csv'
    data = pd.read_csv(f'{data_folder_path}/{file_name}', index_col=0)
    fig = subplots.make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.1)
    data['log_income'] = np.log(data['income'])
    data.dropna(inplace=True)
    x_axis_names = {'noise_estimate': 'Noise (dB)', 'log_income': 'Log Median Income', 'age': 'Average Age'}
    for i, x_axis in enumerate(x_axis_names):
        heatmap, line = plotting.heatmap.simple_heat_map_with_mean_value(data=data, x_axis=x_axis, y_axis='bed_time_index', outlier_threshold=0.99, line_width=5, bins=10, line_color='red', zmin=0, zmax=900, showscale=False)
        fig.add_traces([heatmap, line], rows=1, cols=i + 1)
        fig.update_xaxes(title=x_axis_names[x_axis], row=1, col=i + 1)
        if i == 0:
            # tickvals = fig.data[0].y
            # ticktext = [float_to_time(x) for x in fig.data[0].y]
            fig.update_yaxes(title='Night Screen Index', row=1, col=i + 1)
            print(fig.layout.yaxis.tickvals)

    # fig.add_annotation(x=0.05, y=0.95, xref='paper', yref='paper', text=f'R<sup>2</sup> = {reg.rsquared:.2f}  Slope = {np.round(reg.params[1], decimals=4)}', showarrow=False, font=dict(size=30, color="Black", family="Arial"))
    fig.update_layout(template='plotly_white', font=dict(size=30, color="Black", family="Arial"), width=1500, height=600)
    colorbar_trace = go.Scatter(x=[None], y=[None],mode='markers',
                                marker=dict(colorscale='Greys', showscale=True, cmin=0, cmax=800, colorbar=dict(thickness=30, title=dict(text='Number of observations', side='right'), ypad=10, outlinewidth=0)), showlegend=False)
    fig.add_trace(colorbar_trace)
    fig.show()

    save_plotly_fig(fig, 'regression_plot_abs')


def float_to_time(x):
    from datetime import datetime
    ts = pd.date_range(start=datetime(2019, 1, 1, 21), end=datetime(2019, 1, 2, 3, 30), freq='1min')
    times = pd.date_range(start=datetime(2019, 1, 1, 21), end=datetime(2019, 1, 2, 3, 30), freq='15min')
    multiplier = len(ts) / len(times)
    index = np.round(x * multiplier).astype(int)
    return ts[index].time().strftime('%H:%M')

def matching_plot():
    file_name = 'matching_figure.csv'
    data = pd.read_csv(f'{data_folder_path}/{file_name}', index_col=0)

    val, ci, nobs = get_bar_value_and_conf_int(data=data, var='bed_time_index')

    d_eqip = data.loc[(data['night_equipment_counts_high'] < 3) & (data['night_equipment_counts_low'] < 3)].copy()
    val_eqip, ci_equip, nobs_eqip = get_bar_value_and_conf_int(data=d_eqip, var='bed_time_index')

    d_pop = data.loc[(data['pop_high'] > 100) & (data['pop_low'] > 100)].copy()
    val_pop, ci_pop, nobs_pop = get_bar_value_and_conf_int(data=d_pop, var='bed_time_index')

    d_income = select_observations_with_diff_below_threshold(data=data, var='income', threshold=5000)
    d_income_age = select_observations_with_diff_below_threshold(data=d_income, var='age', threshold=3)
    val_income_age, ci_income_age, nobs_income_age = get_bar_value_and_conf_int(data=d_income_age, var='bed_time_index')

    fig = go.Figure()
    fig.add_trace(go.Bar(x=['No controls', 'Services', 'Population', 'Income + Age'], y=[val, val_eqip, val_pop, val_income_age], error_y=dict(type='data', array=[ci, ci_equip, ci_pop, ci_income_age], visible=True, color='red'), marker_color='black', showlegend=False))
    fig.update_layout(template='plotly_white', font=dict(size=30, color="Black", family="Arial"), width=1500, height=600, yaxis_title='Pct change in Night Screen Index', xaxis_title='Controls')
    print(f'Number of observations: {nobs}, {nobs_eqip}, {nobs_pop}, {nobs_income_age}')
    fig.show()
    save_plotly_fig(fig, 'matching_plot_abs')


def get_bar_value_and_conf_int(data: pd.DataFrame, var: str, outlier_threshold: float = 0.99):
    rel_diff = ((data[f'{var}_high'] - data[f'{var}_low']) / data[f'{var}_low']).to_frame(name=var)
    mask_outliers = (rel_diff[var] > rel_diff[var].quantile(outlier_threshold)) | (rel_diff[var] < rel_diff[var].quantile(1 - outlier_threshold))
    rel_diff = rel_diff.loc[~mask_outliers][var]
    mean_diff = 100* rel_diff.mean()
    std_diff = 100* rel_diff.std()
    n_obs = len(rel_diff)
    conf_int = 1.96 * std_diff / np.sqrt(n_obs)
    return mean_diff, conf_int, n_obs


def select_observations_with_diff_below_threshold(data: pd.DataFrame, var: str, threshold: float):
    var_diff = data[f'{var}_high'] - data[f'{var}_low']
    data = data.loc[var_diff.abs() < threshold]
    return data


if __name__ == '__main__':
    # paris_map_plots()
    # regression_plots()
    matching_plot()