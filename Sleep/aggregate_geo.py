from typing import List, Callable, Dict, Union
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import insee
import mobile_traffic as mt

from mobile_data import MobileData, TrafficData, ScreenTimeData


# Matchings
# ---------

def get_matching_netmob_tile_to_insee_tile(city_netmob_tile: mt.City) -> pd.DataFrame:
    city_netmob_tile_geo = mt.geo_tile.get_geo_data(city=city_netmob_tile)
    matching_netmob_tile_to_insee_tile = insee.tile.get_matching_tiles(polygons=city_netmob_tile_geo, return_intersection_area=True)
    matching_netmob_tile_to_insee_tile.rename(columns={'tile': 'insee_tile'}, inplace=True)
    return _clean_matching(matching=matching_netmob_tile_to_insee_tile, index_name='netmob_tile', column_name='insee_tile')


def get_matching_netmob_tile_to_iris(city_netmob_tile: mt.City) -> pd.DataFrame:
    city_netmob_tile_geo = mt.geo_tile.get_geo_data(city=city_netmob_tile)
    matching_netmob_tile_to_iris = insee.iris.geo.get_matching_iris(polygons=city_netmob_tile_geo, return_intersection_area=True)
    matching_netmob_tile_to_iris.rename(columns={'iris': 'insee_iris'}, inplace=True)
    return _clean_matching(matching=matching_netmob_tile_to_iris, index_name='netmob_tile', column_name='insee_iris')


def get_matching_netmob_tile_to_commune(city_netmob_tile: mt.City) -> pd.DataFrame:
    city_netmob_tile_geo = mt.geo_tile.get_geo_data(city=city_netmob_tile)
    matching_netmob_tile_to_commune = insee.commune.get_matching_communes(polygons=city_netmob_tile_geo, with_arrondissement_municipal=True, return_intersection_area=True)
    matching_netmob_tile_to_commune.rename(columns={'commune': 'insee_com'}, inplace=True)
    return _clean_matching(matching=matching_netmob_tile_to_commune, index_name='netmob_tile', column_name='insee_com')


def get_matching_iris_to_commune() -> pd.DataFrame:
    iris = insee.iris.geo.get_geo_data()
    matching_iris_commune = insee.commune.get_matching_communes(polygons=iris, with_arrondissement_municipal=True, return_intersection_area=True)
    return _clean_matching(matching=matching_iris_commune, index_name='iris', column_name='commune')


def _clean_matching(matching: pd.DataFrame, index_name: str, column_name: str):
    matching.reset_index(names=[index_name], inplace=True)
    matching = matching[[index_name, column_name, 'intersection_area']].copy()
    matching.dropna(inplace=True)
    return matching


# 2D aggregation
# ---------------

def aggregate_netmob_tile_level_variables_to_insee_tile_level(data: pd.DataFrame, city_netmob_tile: mt.City, variables: List[str], aggregation_fct: Union[str, Dict[str, Union[Callable, str]]]) -> pd.DataFrame:
    matching_netmob_tile_to_insee_tile = get_matching_netmob_tile_to_insee_tile(city_netmob_tile=city_netmob_tile).set_index('netmob_tile')
    data_ = _merge_data_and_matching(data=data, matching=matching_netmob_tile_to_insee_tile, column_drop_na='insee_tile')
    data_ = perform_aggregation(data=data_, variables=variables, aggregation_fct=aggregation_fct, group_by_var='insee_tile')
    data_ = _add_covered_area_and_reset_index(data=data_, area_geo_data=insee.tile.get_geo_data(tile=data_.index).to_crs(epsg=2154).area)
    return data_


def aggregate_netmob_tile_level_variables_to_iris_level(data: pd.DataFrame, city_netmob_tile: mt.City, variables: List[str], aggregation_fct: Union[str, Dict[str, Union[Callable, str]]]) -> pd.DataFrame:
    matching_netmob_tile_to_iris = get_matching_netmob_tile_to_iris(city_netmob_tile=city_netmob_tile).set_index('netmob_tile')
    data_ = _merge_data_and_matching(data=data, matching=matching_netmob_tile_to_iris, column_drop_na='insee_iris')
    data_ = perform_aggregation(data=data_, variables=variables, aggregation_fct=aggregation_fct, group_by_var='insee_iris')
    data_ = _add_covered_area_and_reset_index(data=data_, area_geo_data=insee.iris.geo.get_geo_data(iris=data_.index).to_crs(epsg=2154).area)
    return data_


def aggregate_netmob_tile_level_variables_to_commune_level(data: pd.DataFrame, city_netmob_tile: mt.City, variables: List[str], aggregation_fct: Union[str, Dict[str, Union[Callable, str]]]) -> pd.DataFrame:
    matching_netmob_tile_to_commune = get_matching_netmob_tile_to_commune(city_netmob_tile=city_netmob_tile).set_index('netmob_tile')
    data_ = _merge_data_and_matching(data=data, matching=matching_netmob_tile_to_commune, column_drop_na='insee_com')
    data_ = perform_aggregation(data=data_, variables=variables, aggregation_fct=aggregation_fct, group_by_var='insee_com')
    data_ = _add_covered_area_and_reset_index(data=data_, area_geo_data=insee.commune.get_geo_data(commune_ids=data_.index, with_arrondissement_municipal=True).to_crs(epsg=2154).area)
    return data_


def aggregate_iris_level_variables_to_commune_level(data: pd.DataFrame, variables: List[str], aggregation_fct: Dict[str, Union[Callable, str]] = None) -> pd.DataFrame:
    matching_iris_commune = get_matching_iris_to_commune().set_index('iris')
    data_ = _merge_data_and_matching(data=data, matching=matching_iris_commune, column_drop_na='commune')
    data_ = perform_aggregation(data=data_, variables=variables, aggregation_fct=aggregation_fct, group_by_var='commune')
    data_ = _add_covered_area_and_reset_index(data=data_, area_geo_data=insee.commune.get_geo_data(commune_ids=data_.index, with_arrondissement_municipal=True).to_crs(epsg=2154).area)
    return data_


def _merge_data_and_matching(data: pd.DataFrame, matching: pd.DataFrame, column_drop_na: str) -> pd.DataFrame:
    data_ = pd.merge(data, matching, left_index=True, right_index=True)
    data_.reset_index(inplace=True)
    data_.dropna(subset=[column_drop_na], inplace=True)
    return data_


def _add_covered_area_and_reset_index(data: pd.DataFrame, area_geo_data: pd.DataFrame) -> pd.DataFrame:
    data['covered_area'] = data['covered_area'] / area_geo_data
    data.reset_index(inplace=True)
    return data


def perform_aggregation(data: pd.DataFrame, variables: List[str], aggregation_fct: Union[str, Dict[str, Union[Callable, str]]], group_by_var: str) -> pd.DataFrame:
    def weighted_mean(x):
        area = data.loc[x.index, 'intersection_area'].values
        return np.average(x.values, weights=area)

    def weighted_sum(x):
        area_share = data.loc[x.index, 'intersection_area'].values / (100 ** 2)
        return np.sum(x.values * area_share)

    aggregation_fct_ = {v: aggregation_fct for v in variables} if isinstance(aggregation_fct, str) else aggregation_fct
    for v, fct in aggregation_fct_.items():
        if fct == 'weighted_mean':
            aggregation_fct_[v] = weighted_mean
        elif fct == 'weighted_sum':
            aggregation_fct_[v] = weighted_sum

    aggregation_fct_['intersection_area'] = "sum"
    data_ = data.groupby(group_by_var).agg(aggregation_fct_)
    data_.rename(columns={'intersection_area': 'covered_area'}, inplace=True)
    return data_

# 3D aggregation
# ---------------


def aggregate_netmob_tile_level_traffic_data_to_insee_tile_level(traffic_data_netmob_tile: TrafficData) -> TrafficData:
    traffic_data_insee_tile = {city: aggregate_netmob_tile_level_traffic_data_city_to_insee_tile_level(traffic_data_netmob_tile=traffic_data_city, city=city) for city, traffic_data_city in tqdm(traffic_data_netmob_tile.data.items())}
    return TrafficData(data=traffic_data_insee_tile)


def aggregate_netmob_tile_level_traffic_data_city_to_insee_tile_level(traffic_data_netmob_tile: xr.DataArray, city: mt.City) -> xr.DataArray:
    times = traffic_data_netmob_tile.time.values
    services = traffic_data_netmob_tile.service.values
    traffic_data_insee_tiles = []
    for t in tqdm(times):
        traffic_data_netmob_tile_time_t = traffic_data_netmob_tile.sel(time=t).to_pandas()
        traffic_data_insee_tile_time_t = aggregate_netmob_tile_level_variables_to_insee_tile_level(data=traffic_data_netmob_tile_time_t, city_netmob_tile=city, variables=services, aggregation_fct='weighted_sum')
        traffic_data_insee_tile_time_t.set_index('insee_tile', inplace=True)
        traffic_data_insee_tile_time_t = traffic_data_insee_tile_time_t.loc[traffic_data_insee_tile_time_t['covered_area'] > 0.8].copy()
        traffic_data_insee_tile_time_t.drop(columns=['covered_area'], inplace=True)
        traffic_data_insee_tiles.append(traffic_data_insee_tile_time_t)

    insee_tiles = traffic_data_insee_tiles[0].index
    traffic_data_insee_tiles = np.stack([d.values for d in traffic_data_insee_tiles], axis=-1)
    dims = ['insee_tile', 'service', 'time']
    coords = [insee_tiles, services, times]
    traffic_data_insee_tiles = xr.DataArray(traffic_data_insee_tiles, dims=dims, coords=coords)
    return traffic_data_insee_tiles
