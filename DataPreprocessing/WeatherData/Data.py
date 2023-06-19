import os
from datetime import date, time

import pandas as pd

from config import WEATHER_DATA_PATH


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
        data.sort_values(by=['date'], inplace=True)
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
        data.sort_values(by=['day'], inplace=True)
        return data
