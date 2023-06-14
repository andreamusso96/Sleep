import pandas as pd

from DataPreprocessing.GeoData.GeoData import IrisGeoData, TileGeoData, WeatherStationGeoData
from DataPreprocessing.GeoData.GeoMatchers.IrisTileMatcher import IrisTileMatcher
from DataPreprocessing.GeoData.GeoMatchers.IrisWeatherStationMatcher import IrisWeatherStationMatcher
from Utils import City, AggregationLevel
from config import GEO_DATA_PATH


class GeoMatching:
    def __init__(self, file_name: str):
        self.file_path = f'{GEO_DATA_PATH}/{file_name}'
        self.data = self.load()
        self.mapping_from_city_tile_to_iris = self._mapping_from_city_tile_to_iris()
        self.mapping_from_iris_to_weather_station = self._mapping_from_iris_to_weather_station()
        self.mapping_from_iris_to_city = self._mapping_from_iris_to_city()
        self.mapping_city_to_weather_station = self._mapping_city_to_weather_station()

    def load(self):
        data = pd.read_csv(filepath_or_buffer=self.file_path, sep=',',
                           dtype={AggregationLevel.TILE.value: int, AggregationLevel.IRIS.value: str})
        return data

    def _mapping_from_city_tile_to_iris(self):
        mapping = {}
        for city in City:
            city_matching_data = self.data[self.data['city'] == city.value]
            mapping[city.value] = {tile_id: iris_id for tile_id, iris_id in
                                   zip(city_matching_data[AggregationLevel.TILE.value],
                                       city_matching_data[AggregationLevel.IRIS.value])}
        return mapping

    def _mapping_from_iris_to_weather_station(self):
        weather_station_iris_matching_data = self.data.drop_duplicates(subset=[AggregationLevel.IRIS.value])[[AggregationLevel.IRIS.value, 'weather_station']]
        mapping = {iris: weather_station for iris, weather_station in weather_station_iris_matching_data.values}
        return mapping

    def _mapping_from_iris_to_city(self):
        iris_city_matching_data = self.data.drop_duplicates(subset=[AggregationLevel.IRIS.value])[[AggregationLevel.IRIS.value, 'city']]
        mapping = {iris: city for iris, city in iris_city_matching_data.values}
        return mapping

    def _mapping_city_to_weather_station(self):
        city_weather_station_matching_data = self.data.drop_duplicates(subset=['city'])[['city', 'weather_station']]
        mapping = {city: weather_station for city, weather_station in city_weather_station_matching_data.values}
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


class GeoMatchingAPI:
    matching_file_name = f'geo_matching.csv'
    @staticmethod
    def generate_matching_file():
        iris_geo_data = IrisGeoData()
        iris_tile_matching = GeoMatchingAPI._get_iris_tile_matching(iris_geo_data=iris_geo_data)
        iris_weather_station_matching = GeoMatchingAPI._get_iris_weather_station_matching(iris_geo_data=iris_geo_data)
        matching = iris_tile_matching.merge(iris_weather_station_matching, on='iris', how='left')
        matching.to_csv(path_or_buf=f'{GEO_DATA_PATH}/{GeoMatchingAPI.matching_file_name}', index=False)

    @staticmethod
    def _get_iris_tile_matching(iris_geo_data: IrisGeoData):
        tile_geo_data = [TileGeoData(city=city) for city in City]
        iris_tile_matcher = IrisTileMatcher(tile_geo_data=tile_geo_data, iris_geo_data=iris_geo_data)
        matching = iris_tile_matcher.get_matching()
        return matching

    @staticmethod
    def _get_iris_weather_station_matching(iris_geo_data: IrisGeoData):
        weather_station_geo_data = WeatherStationGeoData()
        iris_weather_station_matcher = IrisWeatherStationMatcher(iris_geo_data=iris_geo_data, weather_station_geo_data=weather_station_geo_data)
        matching = iris_weather_station_matcher.get_matching()
        return matching

    @staticmethod
    def load_matching() -> GeoMatching:
        geo_matching = GeoMatching(file_name=GeoMatchingAPI.matching_file_name)
        return geo_matching


if __name__ == '__main__':
    m = GeoMatchingAPI.load_matching()
    m = m.get_weather_station(city=City.LYON)
