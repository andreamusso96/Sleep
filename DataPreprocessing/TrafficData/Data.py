from datetime import time, date, datetime, timedelta
from typing import List
import itertools

import pandas as pd
import xarray as xr
import numpy as np

from Utils import City, TrafficType, TrafficDataDimensions, Calendar, Anomalies
from DataIO import DataIO
from DataPreprocessing.GeoData.GeoDataType import GeoDataType


class CityTrafficData:
    def __init__(self, city: City, geo_data_type: GeoDataType = GeoDataType.IRIS,
                 traffic_type: TrafficType = TrafficType.USERS):
        self.city = city
        self.aggregation_level = geo_data_type
        self.traffic_type = traffic_type
        self.data = self.load()
        self.data_datetime_index = None

    def load(self):
        city_traffic_data = DataIO.load_traffic_data(traffic_type=self.traffic_type,
                                                     geo_data_type=self.aggregation_level, city=self.city)
        return city_traffic_data

    def get_traffic_time_series_by_location(self, remove_nights_before_holidays: bool = True, remove_nights_of_anomaly_periods: bool = True) -> pd.DataFrame:
        traffic_time_series_by_location = self.day_time_to_datetime_index(xar=self.data.sum(dim=TrafficDataDimensions.SERVICE.value)).T.to_pandas()
        if remove_nights_before_holidays:
            traffic_time_series_by_location = self._remove_fridays_and_saturdays(traffic_time_series=traffic_time_series_by_location)
            traffic_time_series_by_location = self._remove_nights_before_holidays(traffic_time_series=traffic_time_series_by_location)
        if remove_nights_of_anomaly_periods:
            traffic_time_series_by_location = self._remove_nights_of_anomaly_periods(traffic_time_series=traffic_time_series_by_location)
            traffic_time_series_by_location = self._drop_first_day(traffic_time_series=traffic_time_series_by_location)  # Since the first day is a saturday, we cut of its night. If we do not remove it, we have half a day detached from the rest of our series.
        return traffic_time_series_by_location

    def _remove_fridays_and_saturdays(self, traffic_time_series: pd.DataFrame) -> pd.DataFrame:
        traffic_time_series_no_fridays_and_saturdays = self._remove_days(traffic_time_series=traffic_time_series, days=Calendar.fridays_and_saturdays())
        return traffic_time_series_no_fridays_and_saturdays

    def _remove_nights_before_holidays(self, traffic_time_series: pd.DataFrame) -> pd.DataFrame:
        holidays = Calendar.holidays()
        nights_before_holidays = [holiday - timedelta(days=1) for holiday in holidays]
        traffic_time_series_no_holidays = self._remove_days(traffic_time_series=traffic_time_series, days=nights_before_holidays)
        return traffic_time_series_no_holidays

    def _remove_nights_of_anomaly_periods(self, traffic_time_series: pd.DataFrame) -> pd.DataFrame:
        traffic_time_series_no_anomalies = self._remove_days(traffic_time_series=traffic_time_series, days=Anomalies.get_anomaly_dates_by_city(city=self.city))
        return traffic_time_series_no_anomalies

    def _drop_first_day(self, traffic_time_series: pd.DataFrame) -> pd.DataFrame:
        traffic_time_series_without_first_and_last_day = self._remove_days(traffic_time_series=traffic_time_series, days=[traffic_time_series.index[0].date()], start_day=time(0))
        return traffic_time_series_without_first_and_last_day

    @staticmethod
    def _remove_days(traffic_time_series: pd.DataFrame, days: List[date], start_day: time = time(15)) -> pd.DataFrame:
        datetime_to_remove = list(itertools.chain(*[CityTrafficData._get_24_day_from_start_time(day=day, start_day=start_day) for day in days]))
        traffic_time_series = traffic_time_series[~traffic_time_series.index.isin(datetime_to_remove)]
        return traffic_time_series

    @staticmethod
    def _get_24_day_from_start_time(day: date, start_day: time):
        datetime_day = [datetime.combine(date=day, time=start_day) + timedelta(minutes=15*i) for i in range(24*4)]
        return datetime_day

    @staticmethod
    def day_time_to_datetime_index(xar: xr.DataArray) -> xr.DataArray:
        new_index = np.add.outer(xar.indexes[TrafficDataDimensions.DAY.value],
                                 xar.indexes[TrafficDataDimensions.TIME.value]).flatten()
        datetime_xar = xar.stack(datetime=[TrafficDataDimensions.DAY.value, TrafficDataDimensions.TIME.value],
                                 create_index=False)
        datetime_xar = datetime_xar.reindex({'datetime': new_index})
        return datetime_xar