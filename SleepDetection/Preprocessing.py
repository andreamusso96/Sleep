from datetime import datetime, time, timedelta

import xarray as xr
import pandas as pd
import numpy as np

from Utils import TrafficDataDimensions


class Smoother:
    def __init__(self):
        self.smoothing_start_time = time(11)
        self.smoothing_end_time = time(19)

    def smooth(self, traffic_time_series_data):
        smoothed_traffic_time_series = traffic_time_series_data.copy()
        mask = (traffic_time_series_data.index.time > self.smoothing_start_time) & (traffic_time_series_data.index.time < self.smoothing_end_time)
        smoothing = traffic_time_series_data.loc[mask].rolling(window=50, center=True, axis=0).median().fillna(method='bfill').fillna(method='ffill')
        smoothed_traffic_time_series.loc[mask] = smoothing
        return smoothed_traffic_time_series


class Chunker:
    def __init__(self):
        self.start_of_day = time(15)

    def chunk_series_into_nights(self, traffic_time_series_data):
        dates = Chunker._get_dates(traffic_time_series_data=traffic_time_series_data)
        nights = [traffic_time_series_data.loc[datetime.combine(date=date, time=self.start_of_day): datetime.combine(date=date, time=self.start_of_day) + timedelta(hours=23, minutes=45)] for date in dates]
        return nights

    @staticmethod
    def _get_dates(traffic_time_series_data):
        return np.unique(traffic_time_series_data.index.date)[:-2]


class Preprocessor:
    def __init__(self, xar_city: xr.DataArray):
        self.xar_city = xar_city

    def preprocess(self) -> pd.DataFrame:
        total_traffic = self.total_traffic()
        traffic_time_series = self.to_time_series_format(total_traffic=total_traffic)
        traffic_time_series = self.index_to_datetime(traffic_time_series=traffic_time_series)
        return traffic_time_series

    def total_traffic(self) -> xr.DataArray:
        return self.xar_city.sum(dim=TrafficDataDimensions.SERVICE.value)

    @staticmethod
    def index_to_datetime(traffic_time_series: pd.DataFrame) -> pd.DataFrame:
        datetime_index = traffic_time_series.index.get_level_values(0) + traffic_time_series.index.get_level_values(1)
        traffic_time_series_dt_index = pd.DataFrame(data=traffic_time_series.values, index=datetime_index, columns=traffic_time_series.columns)
        return traffic_time_series_dt_index

    @staticmethod
    def to_time_series_format(total_traffic: xr.DataArray) -> pd.DataFrame:
        time_series = total_traffic.stack(datetime=[TrafficDataDimensions.DAY.value, TrafficDataDimensions.TIME.value]).T.to_pandas()
        return time_series