from typing import Dict, List, Tuple, Union

import engineer_features as ef
import pandas as pd
import numpy as np
import xarray as xr

import mobile_traffic as mt


def night_screen_index_and_log2_income(screen_time_data: Dict[mt.City, xr.DataArray]) -> pd.DataFrame:
    night_screen_index = ef.night_screen_index_insee_tile(screen_time_data=screen_time_data)
    insee_tiles = list(night_screen_index.index)
    log2_income = ef.log2_mean_income_insee_tile(insee_tiles=insee_tiles)
    nsi_and_income = night_screen_index.merge(log2_income, left_index=True, right_index=True)
    return nsi_and_income


def night_screen_index_and_log2_amenity_counts(screen_time_data: Dict[mt.City, xr.DataArray], buffer_size_m: float = 1000) -> pd.DataFrame:
    night_screen_index = ef.night_screen_index_insee_tile(screen_time_data=screen_time_data)
    insee_tiles = list(night_screen_index.index)
    amenity_count = ef.log2_amenity_counts_insee_tile(insee_tiles=insee_tiles, buffer_size_m=buffer_size_m, frequently_visited_amenities=False)
    frequently_visited_amenities = ef.log2_amenity_counts_insee_tile(insee_tiles=insee_tiles, buffer_size_m=buffer_size_m, frequently_visited_amenities=True)
    nsi_and_amenity_count = night_screen_index.merge(amenity_count, left_index=True, right_index=True)
    nsi_and_amenity_count = nsi_and_amenity_count.merge(frequently_visited_amenities, left_index=True, right_index=True)
    return nsi_and_amenity_count


def traffic_data_by_income_service_time_and_city(traffic_data: Dict[mt.City, xr.DataArray], income_quantiles: List[float]) -> xr.DataArray:
    td_by_income_service_and_time = []
    for city in traffic_data:
        td_city_by_income_service_and_time = group_insee_tile_by_income(traffic_data=traffic_data[city], income_quantiles=income_quantiles)
        td_city_by_income_service_and_time = td_city_by_income_service_and_time.expand_dims(dim={'city': [city]})
        td_by_income_service_and_time.append(td_city_by_income_service_and_time)

    td_by_income_service_time_and_city = xr.concat(td_by_income_service_and_time, dim='city')
    return td_by_income_service_time_and_city


def group_insee_tile_by_income(traffic_data: xr.DataArray, income_quantiles: List[float]):
    insee_tiles = list(traffic_data.insee_tile.values)
    map_insee_tile_to_income_category = get_map_insee_tile_to_income_category(insee_tiles=insee_tiles, income_quantiles=income_quantiles)
    traffic_data = traffic_data.assign_coords(insee_tile=[map_insee_tile_to_income_category[insee_tile] for insee_tile in insee_tiles])
    traffic_data = traffic_data.rename({'insee_tile': 'income_category'})
    traffic_data = traffic_data.groupby('income_category').mean()
    return traffic_data


def get_map_insee_tile_to_income_category(insee_tiles: List[str], income_quantiles: List[float]) -> Dict[str, str]:
    log2_income = ef.log2_mean_income_insee_tile(insee_tiles=insee_tiles)
    log2_income_quantiles = np.quantile(log2_income['log2_income'], q=income_quantiles)
    bins = [-np.inf] + log2_income_quantiles.tolist() + [np.inf]
    income_categories = pd.cut(log2_income['log2_income'], bins=bins, labels=[f'q{k}' for k in range(len(bins) - 1)])
    map_insee_tile_to_income_category = income_categories.to_dict()
    return map_insee_tile_to_income_category


def stack_traffic_data_along_insee_tile_axis(traffic_data: Dict[mt.City, xr.DataArray]) -> xr.DataArray:
    traffic_data_stacked = xr.concat([data for city, data in traffic_data.items()], dim='insee_tile')
    return traffic_data_stacked


def traffic_data_by_income_service_and_time(traffic_data: Dict[mt.City, xr.DataArray], income_quantiles: List[float]) -> xr.DataArray:
    stacked_traffic_data = stack_traffic_data_along_insee_tile_axis(traffic_data=traffic_data)
    td_by_income_service_and_time = group_insee_tile_by_income(traffic_data=stacked_traffic_data, income_quantiles=income_quantiles)
    return td_by_income_service_and_time


def traffic_data_by_income_and_service(traffic_data: Dict[mt.City, xr.DataArray], income_quantiles: List[float]) -> pd.DataFrame:
    td_by_income_service_and_time = traffic_data_by_income_service_and_time(traffic_data=traffic_data, income_quantiles=income_quantiles)
    td_by_income_and_service = td_by_income_service_and_time.sum(dim='time').to_pandas()
    return td_by_income_and_service


def rca_income_and_services(traffic_data: Dict[mt.City, xr.DataArray], income_quantiles: List[float]) -> pd.DataFrame:
    td_by_income_and_service = traffic_data_by_income_and_service(traffic_data=traffic_data, income_quantiles=income_quantiles)
    rca = compute_rca(df=td_by_income_and_service)
    return rca


def compute_rca(df: pd.DataFrame) -> pd.DataFrame:
    numerator = df.div(df.sum(axis=1), axis=0)
    denominator = df.sum(axis=0) / df.sum().sum()
    return numerator.div(denominator, axis=1)


if __name__ == '__main__':
    from synthetic_data import load_synthetic_dataset
    fp = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/SyntheticData'
    td = load_synthetic_dataset(folder_path=fp, insee_tiles=True)
    iq = [0.3, 0.7]
    nsi_log2_income = night_screen_index_and_log2_income(screen_time_data=td)
    nsi_log2_amenity = night_screen_index_and_log2_amenity_counts(screen_time_data=td)
    td_by_istc = traffic_data_by_income_service_time_and_city(traffic_data=td, income_quantiles=iq)
    td_by_ist = traffic_data_by_income_service_and_time(traffic_data=td, income_quantiles=iq)
    td_by_is = traffic_data_by_income_and_service(traffic_data=td, income_quantiles=iq)
    rca_ = rca_income_and_services(traffic_data=td, income_quantiles=iq)
