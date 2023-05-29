from datetime import timedelta, date, time, datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from Logging.Loggers import SleepInferenceLogger


class SleepClassifier:
    def __init__(self, time_series: pd.DataFrame, n_clusters: int, start_day: time, mid_day: time, end_day: time):
        self.time_series = time_series
        self.n_clusters = n_clusters
        self.start_day = start_day
        self.mid_day = mid_day
        self.end_day = end_day
        self.days = np.unique(self.time_series.index.date)[:-1] # TODO: Fix this
        SleepInferenceLogger.debug(f'SleepClassifier: Initialized')

    def make_classification(self) -> pd.DataFrame:
        sleep_scores = []
        for day in self.days:
            sleep_scores_day = self._compute_sleep_scores_day(day=day)
            sleep_scores.append(sleep_scores_day)

        sleep_scores = pd.concat(sleep_scores, axis=0)
        sleep_scores = sleep_scores.sort_index()
        return sleep_scores

    def _compute_sleep_scores_day(self, day: date) -> pd.DataFrame:
        scaler = MinMaxScaler()
        night_traffic = self.time_series.loc[datetime.combine(date=day, time=self.mid_day): datetime.combine(date=day, time=self.mid_day) + timedelta(hours=23, minutes=45)].to_frame()
        night_traffic = pd.DataFrame(scaler.fit_transform(night_traffic).flatten(), index=night_traffic.index, columns=['Traffic'])
        centers = self._get_cluster_centers(day=day, scaler=scaler)
        labels = self._run_clustering_algorithm(centers=centers, night_traffic=night_traffic)
        sleep_scores = labels / (self.n_clusters - 1)
        sleep_scores_and_traffic = pd.concat([sleep_scores, night_traffic], axis=1)
        return sleep_scores_and_traffic

    def _get_cluster_centers(self, day: date, scaler: MinMaxScaler) -> np.ndarray:
        awake_threshold = self._compute_awake_threshold(day=day, scaler=scaler)
        centers = np.linspace(0, awake_threshold, num=self.n_clusters)
        return centers

    def _compute_awake_threshold(self, day: date, scaler: MinMaxScaler) -> float:
        day_before_night_traffic = scaler.transform(self.time_series.loc[datetime.combine(date=day, time=self.start_day): datetime.combine(date=day, time=self.end_day)].to_frame()).flatten()
        day_after_night_traffic = scaler.transform(self.time_series.loc[datetime.combine(date=day + timedelta(days=1), time=self.start_day): datetime.combine(date=day + timedelta(days=1), time=self.end_day)].to_frame()).flatten()
        mean_traffic_surrounding_days = (np.mean(day_before_night_traffic) + np.mean(day_after_night_traffic)) / 2
        awake_threshold = np.linspace(0, mean_traffic_surrounding_days, num=self.n_clusters)[-2]
        return awake_threshold

    def _run_clustering_algorithm(self, centers: np.ndarray, night_traffic: pd.DataFrame) -> pd.DataFrame:
        bins = np.append(centers, 1)
        traffic_vals = night_traffic.values.flatten()
        labels = np.searchsorted(bins, np.where(traffic_vals >= 0.999, 0.999, traffic_vals), side='right') - 1
        labels = pd.DataFrame(data=labels, index=night_traffic.index, columns=['Sleep'])
        return labels