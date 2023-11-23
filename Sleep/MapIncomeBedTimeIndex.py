import pandas as pd
import numpy as np
import geopandas as gpd
import folium

path_data = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/BaseData'
path_figures = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Figures/2023.11.03_Alessandretti'


def load_data():
    bed_time_index = pd.read_csv(f'{path_data}/bed_time_index_insee_tile.csv', index_col=0)
    admin_data = pd.read_csv(f'{path_data}/admin_data_insee_tile.csv', index_col=0)
    log_income = np.log2(admin_data['Ind_snv'] / admin_data['Ind']).to_frame('log_income')
    insee_tile_geo = gpd.read_file(f'{path_data}/insee_tile_geo.geojson').set_index('insee_tile')
    data = pd.merge(bed_time_index, log_income, left_index=True, right_index=True)
    data = pd.merge(data, insee_tile_geo, left_index=True, right_index=True)
    data = gpd.GeoDataFrame(data, geometry='geometry')
    data.dropna(inplace=True)
    return data


def plot_map(data: gpd.GeoDataFrame):
    q_high_bed_time_index = data['bed_time_index'].quantile(0.7)
    q_low_bed_time_index = data['bed_time_index'].quantile(0.3)
    q_low_log_income = data['log_income'].quantile(0.3)
    q_high_low_income = data['log_income'].quantile(0.7)

    high_bed_time_index = data.loc[data['bed_time_index'] >= q_high_bed_time_index].copy()
    low_income = data.loc[data['log_income'] <= q_low_log_income].copy()
    low_bed_time_index = data.loc[data['bed_time_index'] <= q_low_bed_time_index].copy()
    high_income = data.loc[data['log_income'] >= q_high_low_income].copy()

    # paris_map = folium.Map()
    # folium.GeoJson(high_bed_time_index, name='High Night Screen Index').add_to(paris_map)
    # folium.GeoJson(low_income, name='Low Income').add_to(paris_map)

    paris_map = high_bed_time_index.explore(style_kwds=dict(opacity=0.3, color='red'), name='High NightScreen index')
    paris_map = low_income.explore(style_kwds=dict(opacity=0.3, color='blue'), name='Low Income', m=paris_map)
    paris_map = low_bed_time_index.explore(style_kwds=dict(opacity=0.3, color='green'), name='Low NightScreen index', m=paris_map)
    paris_map = high_income.explore(style_kwds=dict(opacity=0.3, color='orange'), name='High Income', m=paris_map)
    folium.LayerControl().add_to(paris_map)
    paris_map.save(f'{path_figures}/paris_map.html')


if __name__ == '__main__':
    d = load_data()
    plot_map(data=d)
    import webbrowser
    webbrowser.open(f'file:///{path_figures}/paris_map.html')

