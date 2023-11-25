from typing import Dict, List, Tuple, Union

import engineer_features as ef
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

import mobile_traffic as mt
import insee
from mobile_data import MobileData, ScreenTimeData, TrafficData
import robustness


# Night screen index
# ------------------

def night_screen_index_for_services(screen_time_data: ScreenTimeData) -> pd.DataFrame:
    night_screen_index = ef.night_screen_index_service(screen_time_data=screen_time_data)
    return night_screen_index


def night_screen_index_and_log2_income(screen_time_data: ScreenTimeData) -> pd.DataFrame:
    night_screen_index = ef.night_screen_index_insee_tile(screen_time_data=screen_time_data)
    insee_tiles = list(night_screen_index.index)
    log2_income = ef.log2_mean_income_insee_tile(insee_tiles=insee_tiles)
    nsi_and_income = night_screen_index.merge(log2_income, left_index=True, right_index=True)
    return nsi_and_income


def night_screen_index_and_log2_income_robustness__screen_time(traffic_data: TrafficData, income_quantiles: List[float], n_samples: int) -> pd.DataFrame:
    nsi_samples = robustness.night_screen_index_samples__screen_time(traffic_data=traffic_data, n_samples=n_samples, traffic_per_minute_sampler=robustness.service_traffic_per_minute_sampler())
    nsi_samples_grouped_by_income = _group_night_screen_index_samples_by_income_category(nsi_samples=nsi_samples, income_quantiles=income_quantiles)
    nsi_samples_mean = nsi_samples_grouped_by_income.mean(axis=1).to_frame('mean')
    nsi_samples_std = nsi_samples_grouped_by_income.std(axis=1).to_frame('std')
    nsi_samples_mean_and_std_per_income_category = pd.concat([nsi_samples_mean, nsi_samples_std], axis=1)
    nsi_samples_mean_and_std_per_income_category.index = nsi_samples_grouped_by_income.index
    return nsi_samples_mean_and_std_per_income_category


def night_screen_index_and_log2_income_robustness__amenity(screen_time_data: ScreenTimeData, income_quantiles: List[float]) -> pd.DataFrame:
    nsi_samples = robustness.night_screen_index_samples__amenity(screen_time_data=screen_time_data, thresholds_and_buffers=robustness.thresholds_and_buffers_amenities())
    nsi_samples_grouped_by_income = _group_night_screen_index_samples_by_income_category(nsi_samples=nsi_samples, income_quantiles=income_quantiles)
    return nsi_samples_grouped_by_income


def _group_night_screen_index_samples_by_income_category(nsi_samples: pd.DataFrame, income_quantiles: List[float]) -> pd.DataFrame:
    map_insee_tile_to_income_category = ef.map_insee_tile_to_income_category(insee_tiles=list(nsi_samples.index), income_quantiles=income_quantiles)
    income_category = pd.Series(map_insee_tile_to_income_category, name='income_category')
    nsi_samples_and_income = nsi_samples.merge(income_category, left_index=True, right_index=True)
    nsi_samples_grouped_by_income = nsi_samples_and_income.groupby('income_category').agg(func=mean_without_nans)
    return nsi_samples_grouped_by_income


def mean_without_nans(x: np.ndarray) -> float:
    if np.isnan(x).all():
        return np.nan
    else:
        return np.nanmean(x)


def night_screen_index_income_category_and_tile_geo(screen_time_data: ScreenTimeData, income_quantiles: List[float]) -> gpd.GeoDataFrame:
    night_screen_index = ef.night_screen_index_insee_tile(screen_time_data=screen_time_data)
    insee_tiles = list(night_screen_index.index)
    map_insee_tile_to_income_category = ef.map_insee_tile_to_income_category(insee_tiles=insee_tiles, income_quantiles=income_quantiles)
    tile_geo = insee.tile.get_geo_data(tile=insee_tiles)
    income_category = pd.Series(map_insee_tile_to_income_category, name='income_category')
    night_screen_index_and_income = night_screen_index.merge(income_category, left_index=True, right_index=True)
    night_screen_index_income_and_tile_geo = night_screen_index_and_income.merge(tile_geo, left_index=True, right_index=True)
    night_screen_index_income_and_tile_geo = gpd.GeoDataFrame(night_screen_index_income_and_tile_geo, geometry='geometry')
    return night_screen_index_income_and_tile_geo


def night_screen_index_and_log2_amenity_counts(screen_time_data: ScreenTimeData, buffer_size_m: float = 1000) -> pd.DataFrame:
    night_screen_index = ef.night_screen_index_insee_tile(screen_time_data=screen_time_data)
    insee_tiles = list(night_screen_index.index)
    amenity_count = ef.log2_amenity_counts_insee_tile(insee_tiles=insee_tiles, buffer_size_m=buffer_size_m, amenity_type=ef.AmenityType.ALL)
    frequently_visited_amenities = ef.log2_amenity_counts_insee_tile(insee_tiles=insee_tiles, buffer_size_m=buffer_size_m, amenity_type=ef.AmenityType.FREQUENTLY_VISITED)
    nsi_and_amenity_count = night_screen_index.merge(amenity_count, left_index=True, right_index=True)
    nsi_and_amenity_count = nsi_and_amenity_count.merge(frequently_visited_amenities, left_index=True, right_index=True)
    return nsi_and_amenity_count

# RCA
# ------------------


def rca_income_and_services(mobile_data: MobileData, income_quantiles: List[float]) -> pd.DataFrame:
    md_by_income_and_service = mobile_data_by_income_and_service(mobile_data=mobile_data, income_quantiles=income_quantiles)
    rca = compute_rca(df=md_by_income_and_service)
    return rca


def rca_income_and_time(screen_time_data: ScreenTimeData, income_quantiles: List[float]) -> pd.DataFrame:
    screen_time_by_income_and_time = screen_time_data_by_income_and_time(screen_time_data=screen_time_data, income_quantiles=income_quantiles)
    rca = compute_rca(df=screen_time_by_income_and_time)
    return rca


def rca_insee_tile_and_service(mobile_data: MobileData) -> pd.DataFrame:
    md_by_insee_tile_and_service = mobile_data_by_insee_tile_and_service(mobile_data=mobile_data)
    rca = compute_rca(df=md_by_insee_tile_and_service)
    return rca


def rca_insee_tile_service_and_tile_geo(mobile_data: MobileData) -> gpd.GeoDataFrame:
    rca = rca_insee_tile_and_service(mobile_data=mobile_data)
    insee_tiles = list(rca.index)
    tile_geo = insee.tile.get_geo_data(tile=insee_tiles)
    rca_and_tile_geo = rca.merge(tile_geo, left_index=True, right_index=True)
    rca_and_tile_geo = gpd.GeoDataFrame(rca_and_tile_geo, geometry='geometry')
    return rca_and_tile_geo


def rca_income_and_services_robustness__screen_time(traffic_data: TrafficData, income_quantiles: List[float], n_samples: int) -> xr.DataArray:
    rca_samples = robustness.rca_income_and_service_samples__screen_time(traffic_data=traffic_data, income_quantiles=income_quantiles, n_samples=n_samples, traffic_per_minute_sampler=robustness.service_traffic_per_minute_sampler())
    return rca_samples


def rca_income_and_services_robustness__amenity(screen_time_data: MobileData, income_quantiles: List[float]) -> xr.DataArray:
    rca_samples = robustness.rca_income_and_service_samples__amenity(mobile_data=screen_time_data, income_quantiles=income_quantiles, thresholds_and_buffers=robustness.thresholds_and_buffers_amenities())
    return rca_samples


def service_space(mobile_data: MobileData) -> pd.DataFrame:
    rca_tile_service = rca_insee_tile_and_service(mobile_data=mobile_data)
    binary_rca = rca_tile_service.map(func=lambda x: 1 if x > 1 else 0)

    co_occurrence_rca = binary_rca.T @ binary_rca
    n_tiles_rca = binary_rca.sum(axis=0)
    max_outer_product_n_tile_rca = pd.DataFrame(np.maximum.outer(n_tiles_rca.values, n_tiles_rca.values), index=n_tiles_rca.index, columns=n_tiles_rca.index)
    proximity = co_occurrence_rca / max_outer_product_n_tile_rca
    return proximity


# General
# ------------------


def mobile_data_by_income_service_time_and_city(mobile_data: MobileData, income_quantiles: List[float]) -> xr.DataArray:
    md_by_income_service_and_time = []
    for city in mobile_data.cities():
        md_city_by_income_service_and_time = group_insee_tile_by_income(data=mobile_data.data[city], income_quantiles=income_quantiles)
        md_city_by_income_service_and_time = md_city_by_income_service_and_time.expand_dims(dim={'city': [city]})
        md_by_income_service_and_time.append(md_city_by_income_service_and_time)

    td_by_income_service_time_and_city = xr.concat(md_by_income_service_and_time, dim='city')
    return td_by_income_service_time_and_city


def group_insee_tile_by_income(data: xr.DataArray, income_quantiles: List[float]):
    insee_tiles = list(data.insee_tile.values)
    map_insee_tile_to_income_category = ef.map_insee_tile_to_income_category(insee_tiles=insee_tiles, income_quantiles=income_quantiles)
    data = data.assign_coords(insee_tile=[map_insee_tile_to_income_category[insee_tile] for insee_tile in insee_tiles])
    data = data.rename({'insee_tile': 'income_category'})
    data = data.groupby('income_category').mean()
    return data


def mobile_data_by_income_service_and_time(mobile_data: MobileData, income_quantiles: List[float]) -> xr.DataArray:
    stacked_mobile_data = mobile_data.stack_data_along_insee_tile_axis()
    md_by_income_service_and_time = group_insee_tile_by_income(data=stacked_mobile_data, income_quantiles=income_quantiles)
    return md_by_income_service_and_time


def mobile_data_by_income_and_service(mobile_data: MobileData, income_quantiles: List[float]) -> pd.DataFrame:
    md_by_income_service_and_time = mobile_data_by_income_service_and_time(mobile_data=mobile_data, income_quantiles=income_quantiles)
    md_by_income_and_service = md_by_income_service_and_time.sum(dim='time').to_pandas()
    return md_by_income_and_service


def screen_time_data_by_income_and_time(screen_time_data: ScreenTimeData, income_quantiles: List[float]) -> pd.DataFrame:
    screen_time_by_income_service_and_time = mobile_data_by_income_service_and_time(mobile_data=screen_time_data, income_quantiles=income_quantiles)
    screen_time_by_income_and_time = screen_time_by_income_service_and_time.sum(dim='service').to_pandas()
    return screen_time_by_income_and_time


def mobile_data_by_insee_tile_and_service(mobile_data: MobileData) -> pd.DataFrame:
    stacked_mobile_data = mobile_data.stack_data_along_insee_tile_axis()
    md_by_insee_tile_and_service = stacked_mobile_data.sum(dim='time').to_pandas()
    return md_by_insee_tile_and_service


def compute_rca(df: pd.DataFrame) -> pd.DataFrame:
    numerator = df.div(df.sum(axis=1), axis=0)
    denominator = df.sum(axis=0) / df.sum().sum()
    return numerator.div(denominator, axis=1)


if __name__ == '__main__':
    from visualize import figure_data_path
    td = MobileData.load_dataset(synthetic=False, insee_tiles=True)
    # td.data = {mt.City.PARIS: td.data[mt.City.PARIS]}
    # s = rca_insee_tile_service_and_tile_geo(mobile_data=td)
    # s.to_file(f'{figure_data_path}/rca_insee_tile_service_and_tile_geo.geojson', driver='GeoJSON')
    s = night_screen_index_income_category_and_tile_geo(screen_time_data=td, income_quantiles=[0.3, 0.7])
    s.to_file(f'{figure_data_path}/nsi_income_tile_geo_trial.geojson', driver='GeoJSON')
