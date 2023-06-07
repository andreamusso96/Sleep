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


if __name__ == '__main__':
    wd = WeatherStationData()
    ssd = SunriseSunsetData()
