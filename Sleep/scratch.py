import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from datetime import datetime, time, timedelta, date
from typing import List

import pandas as pd
import numpy as np
import xarray as xr

import mobile_traffic as mt


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


def plot_map2(data):
    paris_map = data.explore(style_kwds=dict(opacity=1), column='bed_time_index', cmap='RdBu')
    folium.LayerControl().add_to(paris_map)
    paris_map.save(f'{path_figures}/paris_map2.html')


class SessionDistribution:
    def __init__(self, session_distribution: xr.DataArray, start: time, end: time):
        self.data = session_distribution
        self.start = start
        self.end = end
        self.time_index = self.data.coords['time'].values
        auxiliary_day = datetime(2020, 1, 1)
        self.time_index_minutes = [d.time() for d in pd.date_range(start=datetime.combine(date=auxiliary_day, time=self.start), end=datetime.combine(date=auxiliary_day + timedelta(days=1), time=self.end), freq='1min')]

    def probability_of_session_by_time_and_location(self, subset_location: List[str] = None) -> pd.DataFrame:
        data = self.data.sel(iris=subset_location) if subset_location is not None else self.data
        probability = data.mean(dim='day').T.to_pandas()
        return probability

    def expected_time_of_session_by_location(self, subset_location: List[str] = None) -> pd.DataFrame:
        probability = self.probability_of_session_by_time_and_location(subset_location=subset_location)
        expected_time = probability.T.dot(np.arange(len(self.time_index))).to_frame(name='expected_time_number')
        expected_time['expected_time'] = expected_time['expected_time_number'].apply(lambda x: self._float_to_time_in_minutes(float_time=x))
        return expected_time

    def get_deviation_from_mean_of_session_cumulative_distribution_by_location(self):
        probability_by_time_and_location = self.probability_of_session_by_time_and_location()
        cumulative_distribution_across_time = probability_by_time_and_location.cumsum(axis=0)
        mean_cumulative_distribution_across_time = cumulative_distribution_across_time.mean(axis=1)
        deviation_from_mean = -1 * cumulative_distribution_across_time.subtract(mean_cumulative_distribution_across_time, axis=0)
        deviation_from_mean = deviation_from_mean.sum(axis=0).to_frame(name='deviation_from_mean')
        return deviation_from_mean

    def _float_to_time_in_minutes(self, float_time: float) -> time:
        multiplier = (len(self.time_index_minutes) - 1) / (len(self.time_index) - 1)
        time_in_minutes = self.time_index_minutes[int(np.round(float_time * multiplier))]
        return time_in_minutes

    def join(self, other: 'SessionDistribution') -> 'SessionDistribution':
        assert np.array_equal(self.time_index, other.time_index), 'SessionDistribution can only be joined if they have the same time index'
        return SessionDistribution(session_distribution=xr.concat([self.data, other.data], dim='location'), start=self.start, end=self.end)



if __name__ == '__main__':
    from mobile_data import TrafficData
    from engineer_features import night_screen_index_insee_tile

    traffic_data = TrafficData.load_dataset(synthetic=False, insee_tiles=True)
    traffic_data.data = {mt.City.PARIS: traffic_data.data[mt.City.PARIS]}

    traffic_data_old = traffic_data.data[mt.City.PARIS]
    probability_by_time_and_location = traffic_data_old.sum(dim='service').to_pandas().T
    probability_by_time_and_location = probability_by_time_and_location / probability_by_time_and_location.sum(axis=0)
    cumulative_distribution_across_time = probability_by_time_and_location.cumsum(axis=0)
    mean_cumulative_distribution_across_time = cumulative_distribution_across_time.mean(axis=1)
    deviation_from_mean = -1 * cumulative_distribution_across_time.subtract(mean_cumulative_distribution_across_time, axis=0)
    deviation_from_mean = deviation_from_mean.sum(axis=0).to_frame(name='deviation_from_mean')
    night_screen_index_old = deviation_from_mean.rename(columns={'deviation_from_mean': 'old'})

    night_screen_index_new = night_screen_index_insee_tile(screen_time_data=traffic_data)
    night_screen_index_new.rename(columns={'night_screen_index': 'new'}, inplace=True)

    joined = pd.merge(night_screen_index_old, night_screen_index_new, left_index=True, right_index=True)



