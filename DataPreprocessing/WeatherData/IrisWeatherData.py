import numpy as np

from DataPreprocessing.WeatherData.Data import WeatherStationData, SunriseSunsetData
from DataPreprocessing.GeoData.GeoData import IrisGeoData, CityLatLongGeoData
from Utils import City


class IrisDataMatcher:
    def __init__(self):
        self.weather_station = WeatherStationData()
        self.sunrise_sunset = SunriseSunsetData()
        self.iris_geo = IrisGeoData()
        self.city_lat_long = CityLatLongGeoData()

    def load(self):
        self.weather_station.load()
        self.weather_station.load_station_locations()
        self.sunrise_sunset.load()
        self.city_lat_long.load()

    def match_iris_to_sunset_sunrise_data(self):
        iris_city_match = self._match_iris_with_city()
        sunrise_sunset_data = self.sunrise_sunset.data
        matched_data = iris_city_match.merge(sunrise_sunset_data, on='city', how='left')
        matched_data.sort_values(by=['iris', 'day'], inplace=True)
        return matched_data

    def match_iris_to_weather_data(self):
        iris_weather_station_match = self._match_iris_with_weather_station()
        weather_data = self._select_weather_variables()
        matched_data = iris_weather_station_match.merge(weather_data, on='station', how='outer')
        matched_data.sort_values(by=['iris', 'datetime'], inplace=True)
        return matched_data

    def _select_weather_variables(self):
        weather_vars = {'numer_sta': 'station', 'date': 'datetime', 't': 'temperature', 'n': 'cloudiness', 'rr1': 'precipitation_last_1h', 'rr3': 'precipitation_last_3h', 'rr12': 'precipitation_last_12h', 'ww': 'weather_classification'}
        weather_data = self.weather_station.data[list(weather_vars.keys())].copy()
        weather_data.rename(columns=weather_vars, inplace=True)
        return weather_data

    def _match_iris_with_weather_station(self):
        iris_data_proj = self.iris_geo.data.to_crs(epsg=2154)
        weather_station_locations = self.weather_station.station_locations.to_crs(epsg=2154)
        iris_weather_station_match = iris_data_proj.sjoin_nearest(weather_station_locations, how='left', distance_col='distance')
        iris_weather_station_match = iris_weather_station_match.loc[iris_weather_station_match.groupby('iris')['distance'].idxmin()][['iris', 'ID']]
        iris_weather_station_match.rename(columns={'ID': 'station'}, inplace=True)
        iris_weather_station_match['station'] = iris_weather_station_match['station'].astype(np.int64)
        return iris_weather_station_match

    def _match_iris_with_city(self):
        iris_data_proj = self.iris_geo.data.to_crs(epsg=2154)
        city_locations = self.city_lat_long.data.to_crs(epsg=2154)
        city_locations = city_locations.loc[city_locations['city'].isin([city.value for city in City])]
        iris_city_match = iris_data_proj.sjoin_nearest(city_locations, how='left', distance_col='distance')
        iris_city_match = iris_city_match.loc[iris_city_match.groupby('iris')['distance'].idxmin()][['iris', 'city']]
        return iris_city_match


if __name__ == '__main__':
    iris_data_matcher = IrisDataMatcher()
    iris_data_matcher.load()
    matched_data = iris_data_matcher.match_iris_to_weather_data()
