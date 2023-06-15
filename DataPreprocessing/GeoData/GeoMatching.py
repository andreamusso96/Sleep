from typing import List

import pandas as pd

from DataPreprocessing.GeoData.GeoData import IrisGeoData, TileGeoData, WeatherStationGeoData, PollingStationGeoData
from DataPreprocessing.GeoData.GeoMatchers.IrisTileMatcher import IrisTileMatcher
from DataPreprocessing.GeoData.GeoMatchers.IrisWeatherStationMatcher import IrisWeatherStationMatcher
from DataPreprocessing.GeoData.GeoMatchers.IrisPollingStationMatcher import IrisPollingStationMatcher
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
from Utils import City
from config import GEO_DATA_PATH


class GeoMatching:
    def __init__(self, file_name: str, load_mappings: bool):
        self.file_path = f'{GEO_DATA_PATH}/{file_name}'
        self.data = self.load()
        if load_mappings:
            self.mapping_from_city_tile_to_iris = self._mapping_from_city_tile_to_iris()
            self.mapping_from_iris_to_weather_station = self._mapping_from_iris_to_weather_station()
            self.mapping_from_iris_to_city = self._mapping_from_iris_to_city()
            self.mapping_city_to_weather_station = self._mapping_from_city_to_weather_station()
            self.mapping_iris_to_polling_stations = self._mapping_from_iris_to_polling_stations()

    def load(self):
        data = pd.read_csv(filepath_or_buffer=self.file_path, sep=',',
                           dtype={GeoDataType.TILE.value: int, GeoDataType.IRIS.value: str})
        return data

    def _mapping_from_city_tile_to_iris(self):
        mapping = {}
        for city in City:
            city_matching_data = self.data[self.data['city'] == city.value].drop_duplicates(subset=[GeoDataType.TILE.value, GeoDataType.IRIS.value])
            mapping[city.value] = {tile_id: iris_id for tile_id, iris_id in zip(city_matching_data[GeoDataType.TILE.value], city_matching_data[GeoDataType.IRIS.value])}
        return mapping

    def _mapping_from_iris_to_weather_station(self):
        weather_station_iris_matching_data = self.data.drop_duplicates(subset=[GeoDataType.IRIS.value])[[GeoDataType.IRIS.value, GeoDataType.WEATHER_STATION.value]]
        mapping = {iris: weather_station for iris, weather_station in weather_station_iris_matching_data.values}
        return mapping

    def _mapping_from_iris_to_city(self):
        iris_city_matching_data = self.data.drop_duplicates(subset=[GeoDataType.IRIS.value])[[GeoDataType.IRIS.value, GeoDataType.CITY.value]]
        mapping = {iris: city for iris, city in iris_city_matching_data.values}
        return mapping

    def _mapping_from_city_to_weather_station(self):
        city_weather_station_matching_data = self.data.drop_duplicates(subset=[GeoDataType.CITY.value])[[GeoDataType.CITY.value, GeoDataType.WEATHER_STATION.value]]
        mapping = {city: weather_station for city, weather_station in city_weather_station_matching_data.values}
        return mapping

    def _mapping_from_iris_to_polling_stations(self):
        city_polling_station_matching_data = self.data.groupby(by=[GeoDataType.IRIS.value])[GeoDataType.POLLING_STATION.value].agg(list).reset_index()
        mapping = {iris: polling_stations for iris, polling_stations in city_polling_station_matching_data.values}
        return mapping

    def get_iris(self, city: City, tile: int) -> int:
        return self.mapping_from_city_tile_to_iris[city.value][tile]

    def get_weather_station(self, iris: str = None, city: City = None):
        if iris is None and city is None:
            raise ValueError('Either iris or city must be provided')
        if iris is not None and city is not None:
            raise ValueError('Either iris or city must be provided')
        if iris is not None:
            return self.mapping_from_iris_to_weather_station[iris]
        if city is not None:
            return self.mapping_city_to_weather_station[city.value]

    def get_city(self, iris: str):
        return self.mapping_from_iris_to_city[iris]

    def get_location_list(self, city: City, geo_data_type: GeoDataType):
        return sorted(self.data[self.data['city'] == city.value][geo_data_type.value].unique().tolist())

    def get_polling_stations(self, iris: str):
        return self.mapping_iris_to_polling_stations[iris]


class GeoMatchingAPI:
    matching_file_name = f'geo_matching.csv'
    @staticmethod
    def generate_matching_file():
        iris_geo_data = IrisGeoData()
        iris_tile_matching = GeoMatchingAPI._get_iris_tile_matching(iris_geo_data=iris_geo_data)
        iris_weather_station_matching = GeoMatchingAPI._get_iris_weather_station_matching(iris_geo_data=iris_geo_data)
        iris_polling_station_matching = GeoMatchingAPI._get_iris_polling_station_matching(iris_geo_data=iris_geo_data)
        matching = iris_tile_matching.merge(iris_weather_station_matching, on=GeoDataType.IRIS.value, how='left')
        matching = matching.merge(iris_polling_station_matching, on=GeoDataType.IRIS.value, how='left')
        matching.to_csv(path_or_buf=f'{GEO_DATA_PATH}/{GeoMatchingAPI.matching_file_name}', index=False)

    @staticmethod
    def _get_iris_tile_matching(iris_geo_data: IrisGeoData):
        tile_geo_data = [TileGeoData(city=city) for city in City]
        iris_tile_matcher = IrisTileMatcher(tile_geo_data=tile_geo_data, iris_geo_data=iris_geo_data)
        matching = iris_tile_matcher.get_matching()
        # matching = pd.read_csv(filepath_or_buffer=f'{GEO_DATA_PATH}/IrisTileMatching.csv', dtype={GeoDataType.TILE.value: int, GeoDataType.IRIS.value: str})
        return matching

    @staticmethod
    def _get_iris_weather_station_matching(iris_geo_data: IrisGeoData):
        weather_station_geo_data = WeatherStationGeoData()
        iris_weather_station_matcher = IrisWeatherStationMatcher(iris_geo_data=iris_geo_data, weather_station_geo_data=weather_station_geo_data)
        matching = iris_weather_station_matcher.get_matching()
        return matching

    @staticmethod
    def _get_iris_polling_station_matching(iris_geo_data: IrisGeoData):
        polling_station_geo_data = PollingStationGeoData()
        tile_polling_station_matcher = IrisPollingStationMatcher(iris_geo_data=iris_geo_data, polling_station_geo_data=polling_station_geo_data)
        matching = tile_polling_station_matcher.get_matching()
        return matching

    @staticmethod
    def load_matching(load_mappings: bool = True) -> GeoMatching:
        geo_matching = GeoMatching(file_name=GeoMatchingAPI.matching_file_name, load_mappings=load_mappings)
        return geo_matching


if __name__ == '__main__':
    geo_matching = GeoMatchingAPI.generate_matching_file()
