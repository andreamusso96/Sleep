from datetime import time, datetime, timedelta

import pandas as pd
import numpy as np

from DataPreprocessing.TrafficData.Data import CityTrafficData
from ExpectedBedTime.ExpectedBedTime import ExpectedBedTime


class Chunker:
    def __init__(self, chunk_start_time: time, chunk_end_time: time):
        self.start_of_chunk = chunk_start_time
        self.end_of_chunk = chunk_end_time
        if self.start_of_chunk > self.end_of_chunk:
            self.day_difference = timedelta(days=1)
        else:
            self.day_difference = timedelta(days=0)

    def chunk_series(self, traffic_time_series):
        dates = self._get_dates(traffic_time_series=traffic_time_series)
        chunks = [traffic_time_series.loc[datetime.combine(date=date, time=self.start_of_chunk): datetime.combine(date=date + self.day_difference, time=self.end_of_chunk)] for date in dates]
        chunks = [chunk for chunk in chunks if not chunk.empty]
        return chunks

    def _get_dates(self, traffic_time_series: pd.DataFrame):
        chunk_times_mask = (traffic_time_series.index.time >= self.start_of_chunk) | (traffic_time_series.index.time <= self.end_of_chunk)
        chunk_dates = np.unique(traffic_time_series.index.date[chunk_times_mask])
        return chunk_dates


class ExpectedBedTimeCalculator:
    def __init__(self, city_traffic_data: CityTrafficData, start_bed_time: time, window_smoothing: int = 3):
        self.city_traffic_data = city_traffic_data
        self.start_bed_time = start_bed_time
        self.window_smoothing = window_smoothing
        self.end_bed_time = time(7)
        self.traffic_time_series_by_location = self._get_smoothed_traffic_time_series()
        self.time_index, self.time_index_with_day = self._get_time_index()

    def _get_time_index(self):
        auxiliary_day = datetime(2020, 1, 2)
        auxiliary_time_index_with_day = pd.date_range(start=datetime.combine(auxiliary_day, self.start_bed_time),
                                        end=datetime.combine(auxiliary_day + timedelta(days=1), self.end_bed_time),
                                        freq='15min')
        time_index = auxiliary_time_index_with_day.time
        return time_index, auxiliary_time_index_with_day.to_numpy()

    def _get_smoothed_traffic_time_series(self):
        traffic_time_series_by_location_rough = self.city_traffic_data.get_traffic_time_series_by_location()
        traffic_time_series_by_location = traffic_time_series_by_location_rough.rolling(window=3, axis=0, center=True).mean().dropna(axis=0, how='all')
        return traffic_time_series_by_location

    def calculate_expected_bed_time(self):
        chunker = Chunker(chunk_start_time=self.start_bed_time, chunk_end_time=self.end_bed_time)
        nights = chunker.chunk_series(traffic_time_series=self.traffic_time_series_by_location)
        expected_bed_times_float_format = pd.concat([self._calculate_expected_bed_time_night(traffic_night=night) for night in nights], axis=0)
        expected_bed_time_summary = self._summarise_expected_bed_times(expected_bed_times_float_format=expected_bed_times_float_format)
        expected_bed_times = ExpectedBedTime(expected_bed_times=expected_bed_time_summary, time_index=self.time_index)
        return expected_bed_times

    def _summarise_expected_bed_times(self, expected_bed_times_float_format: pd.DataFrame):
        mean_expected_bed_times_float_format = expected_bed_times_float_format.mean(axis=0)
        median_expected_bed_times_float_format = expected_bed_times_float_format.median(axis=0)
        std_expected_bed_times_float_format = expected_bed_times_float_format.std(axis=0)
        mean_expected_bed_times_time_format = self._continuous_index_to_time(float_times=mean_expected_bed_times_float_format, times=self.time_index_with_day)
        median_expected_bed_times_time_format = self._continuous_index_to_time(float_times=median_expected_bed_times_float_format, times=self.time_index_with_day)
        std_expected_bed_times_time_format = np.round(15*std_expected_bed_times_float_format).astype(int)
        summary = pd.DataFrame(data={'mean': mean_expected_bed_times_time_format, 'median': median_expected_bed_times_time_format, 'std': std_expected_bed_times_time_format,
                                     'mean_float': mean_expected_bed_times_float_format, 'median_float': median_expected_bed_times_float_format, 'std_float': std_expected_bed_times_float_format, 'n_obs': expected_bed_times_float_format.shape[0]},
                               index=expected_bed_times_float_format.columns)
        return summary

    def _calculate_expected_bed_time_night(self, traffic_night: pd.DataFrame):
        argmin_night = traffic_night.idxmin(axis=0)
        diff_traffic_night = traffic_night.diff(axis=0)
        bed_time_probabilities = self.get_bed_time_probabilities(diff_traffic_night=diff_traffic_night, argmin_traffic_night=argmin_night)
        expected_bed_times_float_format = bed_time_probabilities.T.dot(np.arange(1, traffic_night.shape[0]))
        expected_bed_times_float_format = pd.DataFrame(data=expected_bed_times_float_format.values.reshape(1, -1), index=[traffic_night.index.date[0]], columns=traffic_night.columns)
        return expected_bed_times_float_format

    @staticmethod
    def get_bed_time_probabilities(diff_traffic_night: pd.DataFrame, argmin_traffic_night: pd.Series):
        mask = (argmin_traffic_night.values.reshape(-1, 1) <= diff_traffic_night.index.values).T
        diff_traffic_night_adjusted = diff_traffic_night.where(~mask, other=0)
        diff_traffic_night_adjusted = (-1*diff_traffic_night_adjusted).clip(lower=0)
        diff_traffic_night_adjusted.dropna(axis=0, how='all', inplace=True)
        probability_of_bed_time = diff_traffic_night_adjusted / diff_traffic_night_adjusted.sum(axis=0)
        return probability_of_bed_time

    @staticmethod
    def _continuous_index_to_time(float_times: pd.DataFrame, times: np.ndarray):
        floors = np.floor(float_times).astype(int)
        minutes_to_add = np.round(15*(float_times - floors), decimals=0).astype(int)
        timedelta_to_add = np.array([timedelta(minutes=minutes) for minutes in minutes_to_add], dtype='timedelta64[ns]')
        time_corresponding_to_index = pd.DatetimeIndex(times[floors] + timedelta_to_add).time
        return time_corresponding_to_index




