import os

import pandas as pd
import geopandas as gpd

from config import WEATHER_DATA_PATH


class WeatherStationData:
    def __init__(self):
        self.folder = f'{WEATHER_DATA_PATH}/WeatherStations'
        self.files = [fname for fname in os.listdir(self.folder) if (not fname.startswith('.') and fname.endswith('.csv'))]
        self.data = None
        self.station_locations = None

    def load(self):
        data = [pd.read_csv(f'{self.folder}/{fname}', sep=';', header=0) for fname in self.files]
        data = pd.concat(data, axis=0, ignore_index=True)
        data.drop(columns=['Unnamed: 59'], inplace=True)
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d%H%M%S')
        self.data = data
        return data

    def load_station_locations(self):
        gdf = gpd.read_file(f'{self.folder}/locations.json')
        self.station_locations = gdf
        return gdf


class SunriseSunsetData:
    def __init__(self):
        self.file = f'{WEATHER_DATA_PATH}/sunrise_sunset.csv'
        self.data = None

    def load(self):
        data = pd.read_csv(self.file, sep=',', header=0)
        self.data = data
        return data


class IrisWeatherData:
    def __init__(self):
        self.weather_data_file = f'{WEATHER_DATA_PATH}/iris_weather_data.csv'
        self.sunset_sunrise_data_file = f'{WEATHER_DATA_PATH}/iris_sunset_sunrise_data.csv'
        self.weather_data = None
        self.sunset_sunrise_data = None

    def load_weather_data(self):
        self.weather_data = pd.read_csv(self.weather_data_file, sep=',', header=0, low_memory=False)
        return self.weather_data

    def load_sunset_sunrise_data(self):
        self.sunset_sunrise_data = pd.read_csv(self.sunset_sunrise_data_file, sep=',', header=0, low_memory=False)
        return self.sunset_sunrise_data

    def load_all(self):
        self.load_weather_data()
        self.load_sunset_sunrise_data()
        return self.weather_data, self.sunset_sunrise_data



if __name__ == '__main__':
    wd = WeatherStationData()
    ssd = SunriseSunsetData()
