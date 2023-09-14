from datetime import datetime, time, timedelta, date
from typing import List

import pandas as pd
import numpy as np
import xarray as xr

import mobile_traffic as mt


class SessionDistribution:
    def __init__(self, session_distribution: xr.DataArray, start: time, end: time):
        self.data = session_distribution
        self.start = start
        self.end = end
        self.time_index = self.data.coords['time'].values

    def probability_of_session_by_time_and_location(self, subset_location: List[str] = None) -> pd.DataFrame:
        data = self.data.sel(iris=subset_location) if subset_location is not None else self.data
        probability = data.mean(dim='day').T.to_pandas()
        return probability

    def expected_time_of_session_by_location(self, subset_location: List[str] = None) -> pd.DataFrame:
        probability = self.probability_of_session_by_time_and_location(subset_location=subset_location)
        expected_integer_time = probability.T.dot(np.arange(len(self.time_index)))
        expected_time = expected_integer_time.apply(lambda x: self.time_index[int(np.round(x))])
        expected_time = expected_time.to_frame(name='expected_time')
        return expected_time

    def join(self, other: 'SessionDistribution') -> 'SessionDistribution':
        assert np.array_equal(self.time_index, other.time_index), 'SessionDistribution can only be joined if they have the same time index'
        return SessionDistribution(session_distribution=xr.concat([self.data, other.data], dim='location'), start=self.start, end=self.end)


def calculate_session_distribution_city(traffic_data: mt.CityTrafficData, start: time, end: time, window_smoothing: int = 3, services: List[mt.Service] = None) -> SessionDistribution:
    traffic_time_series_by_location = traffic_data.get_traffic_time_series_by_location(services=services)
    traffic_time_series_by_location = traffic_time_series_by_location.rolling(window=window_smoothing, axis=0, center=True).mean().dropna(axis=0, how='all')
    daily_intervals = split_time_series_into_daily_intervals(time_series=traffic_time_series_by_location, start=start, end=end)
    session_distributions_intervals = [calculate_session_distribution_interval(interval=interval) for interval in daily_intervals]
    session_distribution_3d_array = format_session_distribution_into_3d_xarray(session_distributions_intervals=session_distributions_intervals)
    session_distribution = SessionDistribution(session_distribution=session_distribution_3d_array, start=start, end=end)
    return session_distribution


def split_time_series_into_daily_intervals(time_series: pd.DataFrame, start: time, end: time) -> List[pd.DataFrame]:
    time_is_within_interval = (time_series.index.time >= start) | (time_series.index.time <= end)
    days = np.unique(time_series.index.date[time_is_within_interval])

    day_difference = timedelta(days=1) if start > end else timedelta(days=0)
    chunks = [time_series.loc[datetime.combine(date=day, time=start): datetime.combine(date=day + day_difference, time=end)] for day in days]
    chunks = [chunk for chunk in chunks if not chunk.empty]
    return chunks


def calculate_session_distribution_interval(interval: pd.DataFrame) -> pd.DataFrame:
    total_sessions_within_interval = interval.sum(axis=0)
    session_distribution = interval / total_sessions_within_interval
    return session_distribution


def format_session_distribution_into_3d_xarray(session_distributions_intervals: List[pd.DataFrame]) -> xr.DataArray:
    session_distribution_data = np.stack([session_distributions_interval.values.T for session_distributions_interval in session_distributions_intervals], axis=-1)
    locations = session_distributions_intervals[0].columns.values
    times = session_distributions_intervals[0].index.time
    days = [session_distributions_interval.index[0].date() for session_distributions_interval in session_distributions_intervals]
    coords = {'location': locations, 'time': times, 'day': days}
    dims = ['location', 'time', 'day']
    session_distribution_data = xr.DataArray(session_distribution_data, coords=coords, dims=dims)
    return session_distribution_data