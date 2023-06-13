from typing import List

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import numpy as np

from Utils import City, AggregationLevel
from DataPreprocessing.GeoData.GeoData import TileGeoData, IrisGeoData
from config import GEO_DATA_PATH


class IrisTileMatcher:
    def __init__(self, tile_geo_data: List[TileGeoData], iris_geo_data: IrisGeoData):
        self.tile_geo_data = tile_geo_data
        self.iris_geo_data = iris_geo_data

    def get_matching(self) -> pd.DataFrame:
        matching = []
        iris_geo_data_gpd = self.iris_geo_data.data.to_crs(epsg=2154)
        for tile_geo_data_city in tqdm(self.tile_geo_data):
            tile_geo_data_city_gpd = tile_geo_data_city.data.to_crs(epsg=2154)
            city_matching = self._get_matching_city(tile_geo_data_city=tile_geo_data_city_gpd, iris_geo_data=iris_geo_data_gpd, city_name=tile_geo_data_city.city.value)
            matching.append(city_matching)

        matching = pd.concat(matching, axis=0, ignore_index=True)
        return matching

    @staticmethod
    def _get_matching_city(tile_geo_data_city: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame, city_name: str) -> pd.DataFrame:
        tile_iris_intersections = IrisTileMatcher._get_tile_iris_intersections_with_intersection_area(tile_geo_data_city=tile_geo_data_city, iris_geo_data=iris_geo_data)
        matching_for_tiles_intersecting_iris = IrisTileMatcher._get_matching_for_tiles_intersecting_iris(tile_iris_intersections=tile_iris_intersections)
        tiles_not_intersecting_any_iris = tile_iris_intersections.loc[tile_iris_intersections['index_right'].isna()][[AggregationLevel.TILE.value, 'geometry']]

        if len(tiles_not_intersecting_any_iris) > 0:
            matching_for_tiles_not_intersecting_iris = IrisTileMatcher._get_matching_for_tiles_not_intersecting_any_iris(tiles_not_intersecting_any_iris=tiles_not_intersecting_any_iris, iris_geo_data=iris_geo_data)
            matching = pd.concat([matching_for_tiles_intersecting_iris, matching_for_tiles_not_intersecting_iris], axis=0, ignore_index=True)
        else:
            matching = matching_for_tiles_intersecting_iris

        matching['city'] = city_name
        return matching

    @staticmethod
    def _get_matching_for_tiles_not_intersecting_any_iris(tiles_not_intersecting_any_iris: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        tile_iris_nearest = tiles_not_intersecting_any_iris.sjoin_nearest(right=iris_geo_data, how='left', distance_col='distance')
        matching_for_tiles_not_intersecting_iris = tile_iris_nearest.loc[tile_iris_nearest.groupby(AggregationLevel.TILE.value)['distance'].idxmin()][[AggregationLevel.TILE.value, AggregationLevel.IRIS.value]]
        return matching_for_tiles_not_intersecting_iris

    @staticmethod
    def _get_matching_for_tiles_intersecting_iris(tile_iris_intersections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        tile_iris_intersections_no_nas = tile_iris_intersections.dropna(subset=[AggregationLevel.IRIS.value])
        matching = tile_iris_intersections_no_nas.loc[tile_iris_intersections_no_nas.groupby(AggregationLevel.TILE.value)['intersection_area'].idxmax()][[AggregationLevel.TILE.value, AggregationLevel.IRIS.value]]
        return matching

    @staticmethod
    def _get_tile_iris_intersections_with_intersection_area(tile_geo_data_city: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        tile_iris_intersections = tile_geo_data_city.sjoin(df=iris_geo_data, how='left', predicate='intersects')
        tile_iris_intersections.reset_index(drop=True, inplace=True)
        def intersection_area(x) -> float:
            if np.isnan(x['index_right']):
                return np.nan
            area = x['geometry'].intersection(iris_geo_data.loc[x['index_right'], 'geometry']).area
            return area

        tile_iris_intersections['intersection_area'] = tile_iris_intersections.apply(lambda x: intersection_area(x=x), axis=1)
        return tile_iris_intersections


class IrisTileMatching:
    def __init__(self, file_name: str):
        self.file_name = f'{GEO_DATA_PATH}/{file_name}'
        self.data = self._load()
        self.mapping = self._build_mapping_from_matching_data()

    def _load(self) -> pd.DataFrame:
        data = pd.read_csv(filepath_or_buffer=self.file_name, sep=',', dtype={AggregationLevel.TILE.value: int, AggregationLevel.IRIS.value: str})
        return data

    def get_iris_code(self, city: City, tile_id: int) -> int:
        return self.mapping[city.value][tile_id]

    def _build_mapping_from_matching_data(self):
        mapping = {}
        for city in City:
            city_matching_data = self.data[self.data['city'] == city.value]
            mapping[city.value] = {tile_id: iris_id for tile_id, iris_id in zip(city_matching_data[AggregationLevel.TILE.value], city_matching_data[AggregationLevel.TILE.value])}
        return mapping


class IrisTileMatchingAPI:
    matching_file_name = f'IrisTileMatching.csv'

    @staticmethod
    def generate_matching_file():
        tile_geo_data = [TileGeoData(city=city) for city in City]
        iris_geo_data = IrisGeoData()
        matcher = Matcher(tile_geo_data=tile_geo_data, iris_geo_data=iris_geo_data)
        matching = matcher.get_matching()
        matching.to_csv(path_or_buf=f'{GEO_DATA_PATH}/{IrisTileMatchingAPI.matching_file_name}', index=False)

    @staticmethod
    def load_matching() -> IrisTileMatching:
        iris_tile_matching = IrisTileMatching(file_name=IrisTileMatchingAPI.matching_file_name)
        return iris_tile_matching


if __name__ == '__main__':
    IrisTileMatchingAPI.generate_matching_file()

