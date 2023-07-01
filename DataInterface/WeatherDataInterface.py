from typing import List, Union
from datetime import datetime

import pandas as pd
import numpy as np

from DataInterface.DataInterface import DataInterface
from DataPreprocessing.WeatherData.Data import SunriseSunsetData, WeatherStationData
from DataInterface.GeoDataInterface import GeoData, GeoDataType
from Utils import City


class WeatherData(DataInterface):
    def __init__(self):
        super().__init__()
        self.sunrise_sunset_data = SunriseSunsetData()
        self.weather_station_data = WeatherStationData()
        self.geo_matching = GeoMatchingAPI.load_matching()

    def get_weather_station_data(self, iris: str = None, city: City = None):
        if iris is not None:
            weather_station_code = self.geo_matching.get_weather_station(iris=iris)
        elif city is not None:
            weather_station_code = self.geo_matching.get_weather_station(city=city)
        else:
            raise ValueError('Either subset or city must be provided.')
        weather_station_data = self.weather_station_data.data[
            self.weather_station_data.data['numer_sta'] == weather_station_code]
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

    def get_sunrise_sunset_data(self, geo_data: GeoData, iris: Union[str, List[str]] = None, city: Union[City, List[City]] = None, aggregate: bool = False):
        assert iris is not None or city is not None, 'Either iris or city must be provided.'
        aggregate_by = GeoDataType.IRIS if iris is not None else GeoDataType.CITY
        if iris is not None:
            if isinstance(iris, str):
                iris = [iris]
            city = geo_data.get_geo_data(geometry=GeoDataType.IRIS.value, subset=iris, other_geo_data_types=[GeoDataType.CITY.value])[GeoDataType.CITY.value].unique()
        else:
            if isinstance(city, City):
                city = [city]
            city = np.array([c.value for c in city])

        sunrise_sunset_data = self.sunrise_sunset_data.data[self.sunrise_sunset_data.data[GeoDataType.CITY.value].isin(city)].copy()
        if aggregate:
            sunrise_sunset_data = self._aggregate_sunrise_sunset_data(data=sunrise_sunset_data, aggregate_by=aggregate_by)
        return sunrise_sunset_data

    @staticmethod
    def map_times_to_clockwise_distance_from_mid_night(times: pd.Series):
        return times.apply(lambda x: x.hour + x.minute / 60)

    @staticmethod
    def _aggregate_sunrise_sunset_data(data: pd.DataFrame, aggregate_by: GeoDataType):
        auxiliary_day = datetime(2020, 1, 2)
        data['sunrise'] = data['sunrise'].apply(lambda x: datetime.combine(auxiliary_day, x))
        data['sunset'] = data['sunset'].apply(lambda x: datetime.combine(auxiliary_day, x))
        aggregated_data = data.groupby(aggregate_by.value).agg({'sunrise': 'mean', 'sunset': 'mean'})
        aggregated_data['sunrise'] = aggregated_data['sunrise'].apply(lambda x: x.time())
        aggregated_data['sunset'] = aggregated_data['sunset'].apply(lambda x: x.time())
        return aggregated_data


