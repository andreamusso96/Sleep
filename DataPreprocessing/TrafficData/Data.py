from datetime import time, date, datetime, timedelta
from typing import List
import itertools

import pandas as pd
import xarray
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
        self.data_dt = CityTrafficData.day_time_to_datetime_index(xar=self.data)

    def load(self):
        city_traffic_data = DataIO.load_traffic_data(traffic_type=self.traffic_type,
                                                     geo_data_type=self.aggregation_level, city=self.city)
        return city_traffic_data

    def get_service_consumption_by_location(self, start: time, end: time, remove_holidays: bool = True, remove_anomaly_periods: bool = True):
        traffic_data = self._remove_periods_where_service_consumption_data_is_noisy(traffic_data=self.data_dt.copy(), city=self.city, start=start, end=end, remove_holidays=remove_holidays, remove_anomaly_periods=remove_anomaly_periods)
        service_consumption_by_location__total_hours = traffic_data.sum(dim=TrafficDataDimensions.DATETIME.value).T.to_pandas() / 4
        return service_consumption_by_location__total_hours

    @staticmethod
    def _remove_periods_where_service_consumption_data_is_noisy(traffic_data: xr.DataArray, city: City, start: time, end: time, remove_holidays: bool, remove_anomaly_periods: bool=True):
        if remove_holidays:
            days_holiday = Calendar.holidays()
            days_to_remove = list(set(days_holiday).union(set(Calendar.fridays_and_saturdays())))
            traffic_data = CityTrafficData._remove_24h_periods(traffic_data=traffic_data, dates=days_to_remove, time_start_period=time(0))
        if remove_anomaly_periods:
            anomaly_periods = Anomalies.get_anomaly_dates_by_city(city=city)
            traffic_data = CityTrafficData._remove_24h_periods(traffic_data=traffic_data, dates=anomaly_periods, time_start_period=time(0))

        days = np.unique(traffic_data.datetime.dt.date.values)
        traffic_data = xr.concat([traffic_data.sel(datetime=slice(datetime.combine(day, start), datetime.combine(day, end))) for day in days], dim='datetime')
        return traffic_data

    def get_traffic_time_series_by_location(self, remove_nights_before_holidays: bool = True, remove_nights_of_anomaly_periods: bool = True) -> pd.DataFrame:
        traffic_data = self._remove_nights_where_traffic_data_is_noisy(traffic_data=self.data_dt.sum(dim=TrafficDataDimensions.SERVICE.value), city=self.city, remove_nights_before_holidays=remove_nights_before_holidays, remove_nights_of_anomaly_periods=remove_nights_of_anomaly_periods)
        traffic_time_series_by_location = traffic_data.T.to_pandas()
        return traffic_time_series_by_location

    @staticmethod
    def _remove_nights_where_traffic_data_is_noisy(traffic_data: xr.DataArray, city: City, remove_nights_before_holidays: bool, remove_nights_of_anomaly_periods: bool) -> xr.DataArray:
        if remove_nights_before_holidays:
            days_holiday = Calendar.holidays()
            days_before_holiday = [holiday - timedelta(days=1) for holiday in days_holiday]
            days_to_remove = list(set(days_before_holiday).union(set(Calendar.fridays_and_saturdays())))
            traffic_data = CityTrafficData._remove_24h_periods(traffic_data=traffic_data, dates=days_to_remove, time_start_period=time(15))
            traffic_data = CityTrafficData._remove_24h_periods(traffic_data=traffic_data, dates=[pd.Timestamp(traffic_data.day[0].values).to_pydatetime()], time_start_period=time(0))  # Since the first day is a saturday, we cut of its night. If we do not remove it, we have half a day detached from the rest of our series.
        if remove_nights_of_anomaly_periods:
            days_anomaly = Anomalies.get_anomaly_dates_by_city(city=city)
            days_before_anomaly = [day - timedelta(days=1) for day in days_anomaly]
            days_to_remove = list(set(days_anomaly).union(set(days_before_anomaly)))
            traffic_data = CityTrafficData._remove_24h_periods(traffic_data=traffic_data, dates=days_to_remove, time_start_period=time(15))
        return traffic_data

    @staticmethod
    def _remove_24h_periods(traffic_data: xr.DataArray, dates: List[date], time_start_period: time) -> xr.DataArray:
        for day in dates:
            traffic_data = CityTrafficData._remove_time_period(traffic_data=traffic_data, start=datetime.combine(day, time_start_period), end=datetime.combine(day, time_start_period) + timedelta(days=1))
        return traffic_data

    @staticmethod
    def _remove_time_period(traffic_data: xr.DataArray, start: datetime, end: datetime) -> xr.DataArray:
        traffic_data = xr.concat([traffic_data.sel(datetime=slice(None, start - timedelta(seconds=1))), traffic_data.sel(datetime=slice(end, None))], dim=TrafficDataDimensions.DATETIME.value)
        return traffic_data

    @staticmethod
    def day_time_to_datetime_index(xar: xr.DataArray) -> xr.DataArray:
        new_index = np.add.outer(xar.indexes[TrafficDataDimensions.DAY.value],
                                 xar.indexes[TrafficDataDimensions.TIME.value]).flatten()
        datetime_xar = xar.stack(datetime=[TrafficDataDimensions.DAY.value, TrafficDataDimensions.TIME.value],
                                 create_index=False)
        datetime_xar = datetime_xar.reindex({TrafficDataDimensions.DATETIME.value: new_index})
        return datetime_xar