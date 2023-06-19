from datetime import time, datetime, timedelta
from typing import List

import pandas as pd
import numpy as np
import xarray as xr

from DataInterface.GeoDataInterface import GeoDataType
from DataInterface.TrafficDataInterface import CityTrafficData
from DataInterface.TrafficDataInterface import TrafficDataDimensions
from FeatureExtraction.SessionDistribution.SessionDistribution import SessionDistribution
from FeatureExtraction.SessionDistribution.Chunker import Chunker


class SessionDistributionCalculator:
    def __init__(self, city_traffic_data: CityTrafficData, start: time, end: time, window_smoothing: int = 3):
        super().__init__()
        self.city_traffic_data = city_traffic_data
        self.start = start
        self.end = end
        self.window_smoothing = window_smoothing
        self.traffic_time_series_by_location = self._get_smoothed_traffic_time_series()
        self.time_index = self._get_time_index()
        self.locations = list(self.traffic_time_series_by_location.columns)

    def _get_time_index(self):
        auxiliary_day = datetime(2020, 1, 2)
        auxiliary_time_index_with_day = pd.date_range(start=datetime.combine(auxiliary_day, self.start),
                                        end=datetime.combine(auxiliary_day + timedelta(days=1), self.end),
                                        freq='15min')
        time_index = list(auxiliary_time_index_with_day.time)
        return time_index

    def _get_smoothed_traffic_time_series(self):
        traffic_time_series_by_location_rough = self.city_traffic_data.get_traffic_time_series_by_location()
        traffic_time_series_by_location = traffic_time_series_by_location_rough.rolling(window=3, axis=0, center=True).mean().dropna(axis=0, how='all')
        return traffic_time_series_by_location

    def calculate_session_distribution(self) -> SessionDistribution:
        chunker = Chunker(chunk_start_time=self.start, chunk_end_time=self.end)
        session_counts_timespans = chunker.chunk_series(traffic_time_series=self.traffic_time_series_by_location)
        session_distributions_timespans = [self._calculate_session_distribution_timespan(session_counts_timespan=session_counts_timespan) for session_counts_timespan in session_counts_timespans]
        session_distribution_data = self._format_session_distribution_into_3d_xarray(session_distributions_timespans=session_distributions_timespans)
        session_distribution = SessionDistribution(session_distribution_data=session_distribution_data, start=self.start, end=self.end)
        return session_distribution

    def _format_session_distribution_into_3d_xarray(self, session_distributions_timespans: List[pd.DataFrame]) -> xr.DataArray:
        session_distribution_data = np.stack([session_distribution_timespan.values.T for session_distribution_timespan in session_distributions_timespans], axis=-1)
        coords = {GeoDataType.IRIS.value: self.locations,
                  TrafficDataDimensions.TIME.value: self.time_index,
                  TrafficDataDimensions.DAY.value: [session_distribution_timespan.index[0].date() for session_distribution_timespan in session_distributions_timespans]}
        dims = [GeoDataType.IRIS.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.DAY.value]
        session_distribution_data = xr.DataArray(session_distribution_data, coords=coords, dims=dims)
        return session_distribution_data

    @staticmethod
    def _calculate_session_distribution_timespan(session_counts_timespan: pd.DataFrame):
        total_number_of_sessions = session_counts_timespan.sum(axis=0)
        session_distribution_timespan = session_counts_timespan / total_number_of_sessions
        return session_distribution_timespan
