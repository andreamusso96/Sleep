import geopandas as gpd
import pandas as pd

from DataPreprocessing.GeoData.GeoData import IrisGeoData, CityLatLongGeoData
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
from Utils import City


class IrisCityMatcher:
    def __init__(self, iris_geo_data: IrisGeoData, city_geo_data: CityLatLongGeoData):
        self.iris_geo_data = iris_geo_data
        self.city_geo_data = city_geo_data

    def get_matching(self) -> pd.DataFrame:
        iris_geo_data_proj = self.iris_geo_data.data.to_crs(epsg=2154)
        city_geo_data_proj = self.city_geo_data.data.to_crs(epsg=2154)
        city_geo_data_proj = self._select_cities_in_net_mob_sample(city_geo_data_proj)
        iris_city_matching = self._spatial_join(city_geo_data=city_geo_data_proj, iris_geo_data=iris_geo_data_proj)
        return iris_city_matching

    @staticmethod
    def _select_cities_in_net_mob_sample(city_geo_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        city_geo_data_in_net_mob_sample = city_geo_data.loc[city_geo_data[GeoDataType.CITY.value].isin([city.value for city in City])]
        return city_geo_data_in_net_mob_sample

    @staticmethod
    def _spatial_join(city_geo_data: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame) -> pd.DataFrame:
        iris_city_matching = iris_geo_data.sjoin_nearest(city_geo_data, how='left', distance_col='distance')
        iris_city_matching = iris_city_matching.loc[iris_city_matching.groupby(GeoDataType.IRIS.value)['distance'].idxmin()][[GeoDataType.IRIS.value, GeoDataType.CITY.value]]
        return iris_city_matching