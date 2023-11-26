from typing import List, Dict

import numpy as np
import plotting
import pandas as pd
import networkx as nx
import geopandas as gpd
import network_utils as nu
import graph_tool.all as gt
import mobile_traffic as mt
import folium
import plotly.express as px
import plotly.graph_objects as go
from ast import literal_eval as make_tuple

figure_data_path = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/FigureData'
figure_path = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Figures/2023.11.27_Alessandretti'


# Service space
# ------------


def load_data_plot_service_space():
    service_space = pd.read_csv(f'{figure_data_path}/service_space.csv', index_col=0)
    rca_income_and_service = pd.read_csv(f'{figure_data_path}/rca_income_and_services.csv', index_col=0)
    log10_service_consumption = pd.read_csv(f'{figure_data_path}/log10_service_consumption.csv', index_col=0)

    applications_to_drop = ['Apple_iCloud', 'Apple_Web_Services', 'Dropbox', 'Waze', 'Google_Maps', 'Apple_iTunes', 'Uber', 'Amazon_Web_Services', 'Microsoft_Web_Services', 'Microsoft_Azure', 'Apple_Siri', 'Google_Web_Services', 'Web_Downloads', 'Web_Ads', 'Microsoft_Store']
    service_space.drop(index=applications_to_drop, columns=applications_to_drop, inplace=True)
    rca_income_and_service.drop(index=['q1'], columns=applications_to_drop, inplace=True)
    log10_service_consumption.drop(index=applications_to_drop, inplace=True)

    return service_space, rca_income_and_service, log10_service_consumption


def plot_service_space():
    service_space, rca_income_and_service, log10_service_consumption = load_data_plot_service_space()

    backbone = filter_edges_disparity(service_space=service_space, alpha=0.16)
    spanning_tree = filter_edges_max_spanning_tree(service_space=service_space)
    filtered_service_space = np.maximum(backbone, spanning_tree)

    colors = get_colors(rca_income_and_service=rca_income_and_service)
    sizes = {service: log10_service_consumption.loc[service].values[0] for service in log10_service_consumption.index}

    network_visualization = plotting.network.plot_network(adjacency=filtered_service_space, color=colors, size=sizes)
    network_visualization.show(f'../temp/service_space.html', local=True)


def get_colors(rca_income_and_service: pd.DataFrame) -> Dict[str, str]:
    colors = {}
    for i, income_category in enumerate(rca_income_and_service.index):
        rca_income_category = rca_income_and_service.loc[income_category]
        colors.update({service: px.colors.qualitative.Plotly[i] for service in rca_income_category.index if rca_income_category[service] > 1})

    for service in rca_income_and_service.columns:
        if service not in colors:
            colors.update({service: 'grey'})

    return colors


def filter_edges_disparity(service_space: pd.DataFrame, alpha: float):
    g = get_graph_tool_network(service_space=service_space)
    result = nu.backboning.get_network_backbone(g=g, method=nu.backboning.BackBoneMethod.DISPARITY, alpha=alpha)
    return build_filtered_service_space_from_backbone_result(backbone_result=result, service_space=service_space)


def filter_edges_max_spanning_tree(service_space: pd.DataFrame):
    g = get_graph_tool_network(service_space=service_space)
    result = nu.backboning.get_network_backbone(g=g, method=nu.backboning.BackBoneMethod.MAXIMUM_SPANNING_TREE)
    return build_filtered_service_space_from_backbone_result(backbone_result=result, service_space=service_space)


def get_graph_tool_network(service_space: pd.DataFrame):
    service_space_ = service_space.copy()
    np.fill_diagonal(service_space_.values, 0)
    g = gt.Graph(directed=False)
    edges = np.transpose(service_space_.values.nonzero())
    g.add_edge_list(edges)
    weights = g.new_edge_property(value_type='double', vals=service_space_.values[service_space_.values.nonzero()])
    g.ep['weight'] = weights
    g.vp['id'] = g.new_vertex_property('int', vals=np.arange(len(service_space_)))
    return g


def build_filtered_service_space_from_backbone_result(backbone_result: nu.backboning.BackBoneResult, service_space: pd.DataFrame):
    adjacency = gt.adjacency(backbone_result.backbone_g, weight=backbone_result.backbone_g.ep['weight']).toarray().astype(float)
    filtered_service_space = pd.DataFrame(adjacency, index=service_space.index, columns=service_space.columns)
    return filtered_service_space


# Maps
# ------------

def plot_map_night_screen_index_and_log2_income(city: List[mt.City]):
    night_screen_index_and_income = gpd.read_file(f'{figure_data_path}/nsi_income_tile_geo.geojson', engine="pyogrio")
    night_screen_index_and_income = night_screen_index_and_income.loc[night_screen_index_and_income['city'].isin([c.value for c in city])]

    high_income = night_screen_index_and_income.loc[night_screen_index_and_income['income_category'] == 'q2']
    folium_map = plotting.map.plot_folium_map(data=high_income, name='high_income', column='income_category', color='red', opacity=0.3, legend=False)
    low_income = night_screen_index_and_income.loc[night_screen_index_and_income['income_category'] == 'q0']
    folium_map = plotting.map.plot_folium_map(data=low_income, name='low_income', column='income_category', color='blue', opacity=0.3, folium_map=folium_map, legend=False)

    qs = night_screen_index_and_income['night_screen_index'].quantile([0.3, 0.7]).values
    high_night_screen_index = night_screen_index_and_income.loc[night_screen_index_and_income['night_screen_index'] > qs[1]]
    folium_map = plotting.map.plot_folium_map(data=high_night_screen_index, name='high_nsi', column='night_screen_index', color='yellow', opacity=0.5, folium_map=folium_map, legend=False)
    low_night_screen_index = night_screen_index_and_income.loc[night_screen_index_and_income['night_screen_index'] < qs[0]]
    folium_map = plotting.map.plot_folium_map(data=low_night_screen_index, name='low_nsi', column='night_screen_index', color='green', opacity=0.5, folium_map=folium_map, legend=False)

    folium.LayerControl().add_to(folium_map)
    folium_map.save(f'../temp/map_night_screen_index_and_log2_income.html')


# Correlation night screen index and log2 income
# ------------


def plot_correlation_night_screen_index_and_log2_income():
    nsi_income = pd.read_csv(f'{figure_data_path}/nsi_income.csv', index_col=0)
    heatmap, line = plotting.heatmap.simple_heat_map_with_mean_value(data=nsi_income, x_axis='log2_income', y_axis='night_screen_index', bins=20, line_color='red', line_width=4, rescale_fct=lambda x: np.sqrt(x))
    fig = go.Figure(data=[heatmap, line])
    fig.update_layout(title='Night Screen Index and Log2 Income', xaxis_title='Log2 Income', yaxis_title='Night Screen Index', font=dict(size=30, color='black'), template='plotly_white')
    fig.show(renderer='browser')
    return fig


def plot_robustness_night_screen_index_and_log2_income__screen_time():
    nsi_income_robustness__screen_time__individual = pd.read_csv(f'{figure_data_path}/nsi_income_robustness__screen_time__individual_sampling.csv', index_col=0)
    nsi_income_robustness__screen_time__mean = pd.read_csv(f'{figure_data_path}/nsi_income_robustness__screen_time__mean_sampling.csv', index_col=0)
    fig = go.Figure()
    trace_ind, error_band_ind = plotting.scatter.line_plot_with_shaded_error_bands(line=nsi_income_robustness__screen_time__individual['mean'], error=nsi_income_robustness__screen_time__individual['std'], line_width=4, line_color=px.colors.qualitative.Plotly[0], error_band_opacity=0.3,
                                                                                   name='individual sampling')
    trace_mean, error_band_mean = plotting.scatter.line_plot_with_shaded_error_bands(line=nsi_income_robustness__screen_time__mean['mean'], error=nsi_income_robustness__screen_time__mean['std'], line_width=4, line_color=px.colors.qualitative.Plotly[1], error_band_opacity=0.3, name='mean sampling')
    fig.add_traces([trace_ind, error_band_ind, trace_mean, error_band_mean])
    fig.update_layout(title='Night Screen Index and Log2 Income Robustness (Screen time)', xaxis_title='Log2 Income', yaxis_title='Night Screen Index', font=dict(size=30, color='black'), template='plotly_white')
    fig.show(renderer='browser')
    return fig


def plot_robustness_night_screen_index_and_log2_income__amenities():
    nsi_income_robustness__amenities = pd.read_csv(f'{figure_data_path}/nsi_income_robustness__amenity.csv', index_col=0)
    traces = []
    for col in nsi_income_robustness__amenities.columns:
        name = make_tuple(col.split('_')[-1])
        name = f'thr(log2)={name[0]}, buf={name[1]}'
        trace = go.Scatter(x=nsi_income_robustness__amenities.index, y=nsi_income_robustness__amenities[col], mode='lines + markers', name=name, line=dict(width=4))
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(title='Night Screen Index and Log2 Income Robustness (Amenities)', xaxis_title='Log2 Income', yaxis_title='Night Screen Index', font=dict(size=30, color='black'), template='plotly_white')
    fig.show(renderer='browser')
    return fig


# RCA of income and services


def plot_rca_income_and_service():
    rca = pd.read_csv(f'{figure_data_path}/rca_income_service.csv', index_col=0)
    rca = rca.T.sort_values(by='q0', ascending=False).T
    rca.columns = [col.split('.')[-1] for col in rca.columns]

    fig = go.Figure()
    heatmap = go.Heatmap(z=rca.values, x=rca.columns, y=rca.index, zmid=1, colorscale='RdBu', colorbar=dict(title='RCA', titleside='right'))
    fig.add_trace(heatmap)
    fig.update_layout(title='RCA of mobile app usage by income category', xaxis_title='Service', yaxis_title='Income category', template='plotly_white', font=dict(size=18, color='black'))
    fig.show(renderer='browser')


def plot_robustness_rca_income_and_service__screen_time():
    rca_income_service_robustness__screen_time__individual = pd.read_csv(f'{figure_data_path}/rca_income_service_robustness__screen_time__individual.csv', index_col=0)
    rca_income_service_robustness__screen_time__mean = pd.read_csv(f'{figure_data_path}/rca_income_service_robustness__screen_time__mean.csv', index_col=0)
    fig = go.Figure()
    trace_ind, error_band_ind = plotting.scatter.line_plot_with_shaded_error_bands(line=rca_income_service_robustness__screen_time__individual['mean'], error=rca_income_service_robustness__screen_time__individual['std'], line_width=4, line_color=px.colors.qualitative.Plotly[0],
                                                                                   error_band_opacity=0.3, name='individual sampling')
    trace_mean, error_band_mean = plotting.scatter.line_plot_with_shaded_error_bands(line=rca_income_service_robustness__screen_time__mean['mean'], error=rca_income_service_robustness__screen_time__mean['std'], line_width=4, line_color=px.colors.qualitative.Plotly[1], error_band_opacity=0.3,
                                                                                     name='mean sampling')
    fig.add_traces([trace_ind, error_band_ind, trace_mean, error_band_mean])
    fig.update_layout(title='RCA of mobile app usage by income category Robustness (Screen time)', xaxis_title='Service', yaxis_title='RCA', font=dict(size=30, color='black'), template='plotly_white')
    fig.show(renderer='browser')
    return fig


# RCA income and time
# ------------

def plot_rca_income_and_time():
    rca = pd.read_csv(f'{figure_data_path}/rca_income_service.csv', index_col=0)
    fig = go.Figure()
    heatmap = go.Heatmap(z=rca.values, x=rca.columns, y=rca.index, zmid=1, colorscale='RdBu', colorbar=dict(title='RCA', titleside='right'))
    fig.add_trace(heatmap)
    fig.update_layout(title='RCA of night screen time by income category', xaxis_title='Time', yaxis_title='Income category', template='plotly_white', font=dict(size=18, color='black'))
    fig.show(renderer='browser')


# NSI for services
# ------------
def plot_nsi_for_services():
    nsi_for_services = pd.read_csv(f'{figure_data_path}/nsi_for_services.csv')
    nsi_for_services['service'] = nsi_for_services['service'].apply(lambda x: x.split('.')[-1])
    nsi_for_services = nsi_for_services.pivot(index='city', columns='service', values='night_screen_index')

    fig = go.Figure()
    heatmap = go.Heatmap(z=nsi_for_services.values, x=nsi_for_services.columns, y=nsi_for_services.index, zmid=0, colorscale='RdBu', colorbar=dict(title='NSI', titleside='right'))
    fig.add_trace(heatmap)
    fig.update_layout(title='Night Screen Index for services', xaxis_title='Service', yaxis_title='City', template='plotly_white', font=dict(size=18, color='black'))
    fig.show(renderer='browser')



def show_html_plot(html_file_path: str):
    import webbrowser
    webbrowser.open(f'file://{html_file_path}')


def plot_map_rca_insee_tile_and_service(services: List[mt.Service], colors: List[str]):
    rca_insee_tile_and_service = gpd.read_file(f'{figure_data_path}/rca_insee_tile_service_and_tile_geo.geojson', engine="pyogrio")
    folium_map = plotting.map.plot_folium_map(data=rca_insee_tile_and_service, column=services[0].value, color=colors[0], opacity=0.3, legend=False)
    for i, service in enumerate(services[1:]):
        rca_service_greater_than_1 = rca_insee_tile_and_service.loc[rca_insee_tile_and_service[service.value] > 1][[service.value, 'geometry']]
        folium_map = plotting.map.plot_folium_map(data=rca_service_greater_than_1, column=service.value, color=colors[i + 1], opacity=0.3, folium_map=folium_map, legend=False)

    folium.LayerControl().add_to(folium_map)
    folium_map.save(f'../temp/map_rca_insee_tile_and_service.html')


def _map():
    services = [mt.Service.EA_GAMES, mt.Service.CLASH_OF_CLANS, mt.Service.MICROSOFT_MAIL, mt.Service.WEB_FOOD]
    plot_map_rca_insee_tile_and_service(services=services, colors=px.colors.qualitative.Plotly[:len(services)])
    show_html_plot(f'/Users/andrea/Desktop/PhD/Projects/Current/NetMob/NetMobCode/temp/map_rca_insee_tile_and_service.html')


def _network():
    plot_service_space()
    show_html_plot(f'/Users/andrea/Desktop/PhD/Projects/Current/NetMob/NetMobCode/temp/service_space.html')


def plot_map3():
    night_screen_index_and_income = gpd.read_file(f'{figure_data_path}/nsi_income_tile_geo_trial.geojson', engine="pyogrio")
    night_screen_index_and_income = night_screen_index_and_income.loc[night_screen_index_and_income['city'].isin([mt.City.PARIS.value])]
    folium_map = night_screen_index_and_income.explore(style_kwds=dict(opacity=1), column='night_screen_index', cmap='RdBu', legend=True)
    folium_map.save(f'../temp/paris_map3.html')
    show_html_plot(f'/Users/andrea/Desktop/PhD/Projects/Current/NetMob/NetMobCode/temp/paris_map3.html')


if __name__ == '__main__':
    # _map()
    # _network()
    # plot_map_night_screen_index_and_log2_income(city=[mt.City.PARIS])
    # show_html_plot(f'/Users/andrea/Desktop/PhD/Projects/Current/NetMob/NetMobCode/temp/map_night_screen_index_and_log2_income.html')
    # plot_map3()
    # plot_correlation_night_screen_index_and_log2_income()
    # plot_map_night_screen_index_and_log2_income(city=[mt.City.PARIS])
    plot_rca_income_and_time()
    plot_nsi_for_services()
