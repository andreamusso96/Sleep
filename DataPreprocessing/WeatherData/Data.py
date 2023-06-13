import os
from datetime import date, time

import pandas as pd

from config import WEATHER_DATA_PATH
from DataPreprocessing.GeoData.GeoMatching import GeoMatchingAPI


class WeatherStationData:
    def __init__(self):
        self.folder = f'{WEATHER_DATA_PATH}/WeatherStationData'
        self.files = [fname for fname in os.listdir(self.folder) if (not fname.startswith('.') and fname.endswith('.csv'))]
        self.data = self.load()

    def load(self):
        data = [pd.read_csv(f'{self.folder}/{fname}', sep=';', header=0) for fname in self.files]
        data = pd.concat(data, axis=0, ignore_index=True)
        data.drop(columns=['Unnamed: 59'], inplace=True)
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d%H%M%S')
        return data


class SunriseSunsetData:
    def __init__(self):
        self.file = f'{WEATHER_DATA_PATH}/sunrise_sunset.csv'
        self.data = self.load()

    def load(self):
        data = pd.read_csv(self.file, sep=',', header=0, parse_dates=True)
        data['day'] = data['day'].apply(lambda x: date.fromisoformat(x))
        data['sunrise'] = data['sunrise'].apply(lambda x: time.fromisoformat(x))
        data['sunset'] = data['sunset'].apply(lambda x: time.fromisoformat(x))
        return data


class WeatherData:
    def __init__(self):
        self.sunrise_sunset_data = SunriseSunsetData()
        self.weather_station_data = WeatherStationData()
        self.geo_matching = GeoMatchingAPI.load_matching()

    def get_weather_station_data(self, iris: str):
        weather_station_code = self.geo_matching.get_weather_station(iris=iris)
        weather_station_data = self.weather_station_data.data[self.weather_station_data.data['numer_sta'] == weather_station_code]
        weather_station_data = self._reformat_weather_station_data(weather_station_data=weather_station_data)
        return weather_station_data
    @staticmethod
    def _reformat_weather_station_data(weather_station_data: pd.DataFrame):
        weather_vars = {'numer_sta': 'station', 'date': 'datetime', 't': 'temperature', 'n': 'cloudiness',
                        'rr1': 'precipitation_last_1h', 'rr3': 'precipitation_last_3h',
                        'rr12': 'precipitation_last_12h', 'ww': 'weather_classification'}
        reformatted_weather_data = weather_station_data[list(weather_vars.keys())].copy()
        reformatted_weather_data.rename(columns=weather_vars, inplace=True)
        return reformatted_weather_data

    def get_sunrise_sunset_data(self, iris: str):
        city = self.geo_matching.get_city(iris=iris)
        sunrise_sunset_data = self.sunrise_sunset_data.data[self.sunrise_sunset_data.data['city'] == city]
        return sunrise_sunset_data


if __name__ == '__main__':
    weather_data = WeatherData()
    print(weather_data.get_weather_station_data(iris='751010101'))
    print(weather_data.get_sunrise_sunset_data(iris='751010101'))
