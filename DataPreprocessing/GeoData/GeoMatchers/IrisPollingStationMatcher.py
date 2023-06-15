from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from DataPreprocessing.GeoData.GeoData import IrisGeoData, PollingStationGeoData
from DataPreprocessing.GeoData.GeoDataType import GeoDataType


class IrisPollingStationMatcher:
    def __init__(self, iris_geo_data: IrisGeoData, polling_station_geo_data: PollingStationGeoData):
        self.iris_geo_data = iris_geo_data
        self.polling_station_geo_data = polling_station_geo_data
        self.radius = 200
        self.multipliers = [1, 2, 3, 4, 5, 8, 12, 15]

    def get_matching(self) -> pd.DataFrame:
        polling_station_geo_data_proj = self.polling_station_geo_data.data.to_crs(epsg=2154)
        iris_geo_data_proj = self.iris_geo_data.data.to_crs(epsg=2154)
        matchings, iris_still_to_match = self._generate_matchings_for_all_radii(iris_geo_data=iris_geo_data_proj, polling_station_geo_data=polling_station_geo_data_proj)
        matching_iris_with_polling_stations_within_radii = pd.concat(matchings, axis=0, ignore_index=True)
        matching_iris_with_closest_polling_station_outside_all_radii = IrisPollingStationMatcher._generate_matching_for_iris_with_closest_polling_stations_outside_all_radii(iris_geo_data=iris_still_to_match, polling_station_geo_data=polling_station_geo_data_proj)
        matching = pd.concat([matching_iris_with_polling_stations_within_radii, matching_iris_with_closest_polling_station_outside_all_radii], axis=0, ignore_index=True)
        return matching

    def _generate_matchings_for_all_radii(self, iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame) -> Tuple[List[pd.DataFrame], gpd.GeoDataFrame]:
        matchings = []
        iris_still_to_match = iris_geo_data.copy()
        for multiplier in self.multipliers:
            radius = self.radius * multiplier
            iris_matched_with_polling_stations_within_radius = IrisPollingStationMatcher._match_iris_with_polling_station_within_radius(iris_geo_data=iris_still_to_match, polling_station_geo_data=polling_station_geo_data, radius=radius)
            mask_iris_not_matched = iris_matched_with_polling_stations_within_radius[GeoDataType.POLLING_STATION.value].isna()
            matching = iris_matched_with_polling_stations_within_radius[~mask_iris_not_matched][[GeoDataType.IRIS.value, GeoDataType.POLLING_STATION.value]].copy()
            matching['radius'] = radius
            matchings.append(matching)
            iris_still_to_match = iris_matched_with_polling_stations_within_radius[mask_iris_not_matched][[GeoDataType.IRIS.value, 'geometry']]

        return matchings, iris_still_to_match

    @staticmethod
    def _generate_matching_for_iris_with_closest_polling_stations_outside_all_radii(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame) -> pd.DataFrame:
        iris_matched_with_closest_polling_station = IrisPollingStationMatcher._match_iris_with_closest_polling_station(iris_geo_data=iris_geo_data, polling_station_geo_data=polling_station_geo_data)
        iris_matched_with_closest_polling_station['radius'] = np.inf
        return iris_matched_with_closest_polling_station

    @staticmethod
    def _match_iris_with_polling_station_within_radius(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame, radius: int) -> gpd.GeoDataFrame:
        iris_geo_data['buffer'] = iris_geo_data.centroid.buffer(radius)
        iris_geo_data.set_geometry('buffer', inplace=True)
        iris_polling_station_matching = iris_geo_data.sjoin(polling_station_geo_data, how='left', predicate='intersects')
        iris_polling_station_matching = iris_polling_station_matching[[GeoDataType.IRIS.value, GeoDataType.POLLING_STATION.value, 'geometry']]
        iris_polling_station_matching.set_geometry('geometry', inplace=True)
        iris_polling_station_matching.reset_index(drop=True, inplace=True)
        return iris_polling_station_matching

    @staticmethod
    def _match_iris_with_closest_polling_station(iris_geo_data: gpd.GeoDataFrame, polling_station_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        iris_geo_data['centroid'] = iris_geo_data.centroid
        iris_geo_data.set_geometry('centroid', inplace=True)
        iris_polling_station_join_nearest = iris_geo_data.sjoin_nearest(polling_station_geo_data, how='left', distance_col='distance')
        matching_for_iris_with_no_near_polling_station = iris_polling_station_join_nearest.loc[iris_polling_station_join_nearest.groupby(GeoDataType.IRIS.value)['distance'].idxmin()][[GeoDataType.IRIS.value, GeoDataType.POLLING_STATION.value]]
        return matching_for_iris_with_no_near_polling_station
