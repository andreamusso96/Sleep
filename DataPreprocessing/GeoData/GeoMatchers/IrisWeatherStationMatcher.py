import geopandas as gpd
import numpy as np
import pandas as pd

from DataPreprocessing.GeoData.GeoData import IrisGeoData, WeatherStationGeoData
from DataPreprocessing.GeoData.GeoDataType import GeoDataType


class IrisWeatherStationMatcher:
    def __init__(self, iris_geo_data: IrisGeoData, weather_station_geo_data: WeatherStationGeoData):
        self.iris_geo_data = iris_geo_data
        self.weather_station_geo_data = weather_station_geo_data

    def get_matching(self) -> pd.DataFrame:
        iris_geo_data_proj = self.iris_geo_data.data.to_crs(epsg=2154)
        weather_station_geo_data_proj = self.weather_station_geo_data.data.to_crs(epsg=2154)
        iris_weather_station_matching = self._spatial_join(weather_station_geo_data=weather_station_geo_data_proj, iris_geo_data=iris_geo_data_proj)
        iris_weather_station_matching = self._reformat(iris_weather_station_matching=iris_weather_station_matching)
        return iris_weather_station_matching

    @staticmethod
    def _spatial_join(weather_station_geo_data: gpd.GeoDataFrame, iris_geo_data: gpd.GeoDataFrame) -> pd.DataFrame:
        iris_weather_station_matching = iris_geo_data.sjoin_nearest(weather_station_geo_data, how='left', distance_col='distance')
        iris_weather_station_matching = iris_weather_station_matching.loc[iris_weather_station_matching.groupby(GeoDataType.IRIS.value)['distance'].idxmin()][[GeoDataType.IRIS.value, 'ID']]
        return iris_weather_station_matching

    @staticmethod
    def _reformat(iris_weather_station_matching: pd.DataFrame) -> pd.DataFrame:
        iris_weather_station_matching.rename(columns={'ID': GeoDataType.WEATHER_STATION.value}, inplace=True)
        iris_weather_station_matching[GeoDataType.WEATHER_STATION.value] = iris_weather_station_matching[GeoDataType.WEATHER_STATION.value].astype(np.int64)
        return iris_weather_station_matching