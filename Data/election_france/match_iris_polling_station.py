from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import Data.geo_iris as gi
from . import geo_polling_station as gp

from . import config


def save_matching() -> None:
    matching = get_matching()
    matching.to_csv(config.get_matching_iris_polling_station_file_path(), index=False)


def get_matching() -> pd.DataFrame:
    multipliers = [0, 1, 2, 3, 4, 5, 8, 12, 15]
    radius = 200

    polling_station_geo_data_proj = gp.get_geo_data().to_crs(epsg=2154).reset_index(names='polling_station')
    iris_geo_data_proj = gi.get_geo_data().to_crs(epsg=2154).reset_index(names='iris')

    matchings, iris_still_to_match = generate_matchings_for_all_radii(iris_geo_data=iris_geo_data_proj, polling_station_geo_data=polling_station_geo_data_proj, multipliers=multipliers, radius=radius)
    matching_iris_with_polling_stations_within_radii = pd.concat(matchings, axis=0, ignore_index=True)
    matching_iris_with_closest_polling_station_outside_all_radii = generate_matching_for_iris_with_closest_polling_stations_outside_all_radii(iris_geo_data=iris_still_to_match, polling_station_geo_data=polling_station_geo_data_proj)
    matching = pd.concat([matching_iris_with_polling_stations_within_radii, matching_iris_with_closest_polling_station_outside_all_radii], axis=0, ignore_index=True)
    return matching


def generate_matchings_for_all_radii(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame, multipliers: List[int], radius: int) -> Tuple[List[pd.DataFrame], gpd.GeoDataFrame]:
    matchings = []
    iris_still_to_match = iris_geo_data.copy()
    for multiplier in multipliers:
        r = radius * multiplier
        iris_matched_with_polling_stations_within_radius = match_iris_with_polling_station_within_radius(iris_geo_data=iris_still_to_match, polling_station_geo_data=polling_station_geo_data, radius=r)
        mask_iris_not_matched = iris_matched_with_polling_stations_within_radius['polling_station'].isna()
        matching = iris_matched_with_polling_stations_within_radius[~mask_iris_not_matched][['iris', 'polling_station']].copy()
        matching['radius'] = r
        matchings.append(matching)
        iris_still_to_match = iris_matched_with_polling_stations_within_radius[mask_iris_not_matched][['iris', 'geometry']].copy()

    return matchings, iris_still_to_match


def generate_matching_for_iris_with_closest_polling_stations_outside_all_radii(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame) -> pd.DataFrame:
    iris_matched_with_closest_polling_station = match_iris_with_closest_polling_station(iris_geo_data=iris_geo_data, polling_station_geo_data=polling_station_geo_data)
    iris_matched_with_closest_polling_station['radius'] = np.inf
    return iris_matched_with_closest_polling_station


def match_iris_with_polling_station_within_radius(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame, radius: int) -> gpd.GeoDataFrame:
    iris_geo_data['buffer'] = iris_geo_data.geometry.buffer(radius)
    iris_geo_data.set_geometry('buffer', inplace=True)
    iris_polling_station_matching = iris_geo_data.sjoin(polling_station_geo_data, how='left', predicate='intersects')
    iris_polling_station_matching = iris_polling_station_matching[['iris', 'polling_station', 'geometry']]
    iris_polling_station_matching.set_geometry('geometry', inplace=True)
    iris_polling_station_matching.reset_index(drop=True, inplace=True)
    return iris_polling_station_matching


def match_iris_with_closest_polling_station(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    iris_geo_data['centroid'] = iris_geo_data.centroid
    iris_geo_data.set_geometry('centroid', inplace=True)
    iris_polling_station_join_nearest = iris_geo_data.sjoin_nearest(polling_station_geo_data, how='left', distance_col='distance')
    matching_for_iris_with_no_near_polling_station = iris_polling_station_join_nearest.loc[iris_polling_station_join_nearest.groupby('iris')['distance'].idxmin()][['iris', 'polling_station']]
    return matching_for_iris_with_no_near_polling_station
