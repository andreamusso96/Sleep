import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import numpy as np

import Data.geo_iris as gi
from . import geo_tile as gt

from .enums import City, GeoDataType
from . import config


class GeoMatching:
    def __init__(self):
        self._matching = None

    def load_matching(self):
        self._matching = pd.read_csv(filepath_or_buffer=config.get_matching_iris_tile_file_path(), sep=',', dtype={'tile': int, 'iris': str})

    def get_location_list(self, city: City, geo_data_type: GeoDataType):
        return sorted(self.matching.loc[self.matching['city'] == city.value][geo_data_type.value].unique().tolist())

    def get_iris(self, city: City, tile: int) -> str:
        return self.matching.loc[(self.matching['city'] == city.value) & (self.matching['tile'] == tile)]['iris'].values[0]

    @property
    def matching(self) -> pd.DataFrame:
        if self._matching is None:
            self.load_matching()
        return self._matching


geo_matching: GeoMatching = GeoMatching()


def save_matching():
    matching = get_matching()
    matching.to_csv(path_or_buf=config.get_matching_iris_tile_file_path(), index=False)


def get_matching() -> pd.DataFrame:
    matching = []
    iris_geo_data_gpd = gi.get_geo_data().to_crs(epsg=2154).reset_index(names='iris')
    for city in tqdm(City):
        tile_geo_data_city_gpd = gt.get_geo_data(city=city).to_crs(epsg=2154).reset_index(names='tile')
        city_matching = get_matching_city(tile_geo_data_city=tile_geo_data_city_gpd, iris_geo_data=iris_geo_data_gpd, city_name=city.value)
        matching.append(city_matching)

    matching = pd.concat(matching, axis=0, ignore_index=True)
    return matching


def get_matching_city(tile_geo_data_city: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame, city_name: str) -> pd.DataFrame:
    tile_iris_intersections = get_tile_iris_intersections_with_intersection_area(tile_geo_data_city=tile_geo_data_city, iris_geo_data=iris_geo_data)
    matching_for_tiles_intersecting_iris = get_matching_for_tiles_intersecting_iris(tile_iris_intersections=tile_iris_intersections)
    tiles_not_intersecting_any_iris = tile_iris_intersections.loc[tile_iris_intersections['index_right'].isna()][['tile', 'geometry']]

    if len(tiles_not_intersecting_any_iris) > 0:
        matching_for_tiles_not_intersecting_iris = get_matching_for_tiles_not_intersecting_any_iris(tiles_not_intersecting_any_iris=tiles_not_intersecting_any_iris, iris_geo_data=iris_geo_data)
        matching = pd.concat([matching_for_tiles_intersecting_iris, matching_for_tiles_not_intersecting_iris], axis=0, ignore_index=True)
    else:
        matching = matching_for_tiles_intersecting_iris

    matching['city'] = city_name
    return matching


def get_tile_iris_intersections_with_intersection_area(tile_geo_data_city: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tile_iris_intersections = tile_geo_data_city.sjoin(df=iris_geo_data, how='left', predicate='intersects')
    tile_iris_intersections.reset_index(drop=True, inplace=True)
    def intersection_area(x) -> float:
        if np.isnan(x['index_right']):
            return np.nan
        area = x['geometry'].intersection(iris_geo_data.loc[x['index_right'], 'geometry']).area
        return area

    tile_iris_intersections['intersection_area'] = tile_iris_intersections.apply(lambda x: intersection_area(x=x), axis=1)
    return tile_iris_intersections


def get_matching_for_tiles_intersecting_iris(tile_iris_intersections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tile_iris_intersections_no_nas = tile_iris_intersections.dropna(subset=['iris'])
    matching = tile_iris_intersections_no_nas.loc[tile_iris_intersections_no_nas.groupby('tile')['intersection_area'].idxmax()][['tile', 'iris']]
    return matching


def get_matching_for_tiles_not_intersecting_any_iris(tiles_not_intersecting_any_iris: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tile_iris_nearest = tiles_not_intersecting_any_iris.sjoin_nearest(right=iris_geo_data, how='left', distance_col='distance')
    matching_for_tiles_not_intersecting_iris = tile_iris_nearest.loc[tile_iris_nearest.groupby('tile')['distance'].idxmin()][['tile', 'iris']]
    return matching_for_tiles_not_intersecting_iris