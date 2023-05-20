from datetime import timedelta, time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from Logging.Loggers import SleepInferenceLogger


class SleepClassifier:
    def __init__(self, data: pd.DataFrame, window: int):
        self.data = data
        assert window % 2 == 1, 'Window must be odd'
        self.window = window
        self.time_window = timedelta(minutes=15 * window)
        self.window_half = window // 2
        SleepInferenceLogger.debug(f'SleepClassifier: Initialized for tile {self.data.columns}')

    def cluster(self) -> pd.DataFrame:
        SleepInferenceLogger.debug(f'SleepClassifier: Clustering data for tile {self.data.columns}')
        SleepInferenceLogger.debug(f'SleepClassifier: Extracting datapoints with window size {self.window}')
        data_datetime_index = self.get_data_with_datetime_index()
        rescaled_data = self.rescale_data(data=data_datetime_index)
        dps = self.generate_datapoints(data=rescaled_data, window=self.window, time_window=self.time_window, window_half=self.window_half)
        centers = self.get_cluster_centers(dps=dps)
        SleepInferenceLogger.debug(f'SleepClassifier: Running k-means clustering')
        labels = self.run_kmeans_clustering(dps=dps, centers=centers)
        SleepInferenceLogger.debug(f'SleepClassifier: Postprocessing labels')
        complete_labels = self.add_back_points_for_which_window_was_to_small(labels=labels, data_with_datetime_index=data_datetime_index)
        complete_labels_no_nas = self.fill_nas(labels=complete_labels)
        labels_with_data_index = self.reset_index_to_data_index(labels=complete_labels_no_nas, data_index=self.data.index)
        SleepInferenceLogger.debug(f'SleepClassifier: Finished clustering data for tile {self.data.columns}')
        return labels_with_data_index

    def get_data_with_datetime_index(self) -> pd.DataFrame:
        datetime_index = self.data.index.get_level_values(0) + self.data.index.get_level_values(1)
        data_with_datetime_index = pd.DataFrame(data=self.data.values, index=datetime_index, columns=self.data.columns)
        return data_with_datetime_index

    @staticmethod
    def rescale_data(data) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(data=scaler.fit_transform(data), index=data.index, columns=data.columns)

    @staticmethod
    def generate_datapoints(data: pd.DataFrame, window: int, time_window: timedelta, window_half: int) -> pd.DataFrame:
        dps = [(p.index[window_half], p.values.flatten()) for p in data.rolling(window=time_window, center=True) if len(p) == window]
        dps = pd.DataFrame(data=[dp[1] for dp in dps], index=[dp[0] for dp in dps])
        return dps

    @staticmethod
    def get_cluster_centers(dps: pd.DataFrame) -> pd.DataFrame:
        center_cluster_awake = dps.loc[dps.index.time == time(15, 0)].mean()
        center_cluster_asleep = dps.loc[dps.index.time == time(3, 0)].mean()
        centers = pd.DataFrame(data=[center_cluster_awake, center_cluster_asleep], index=['Awake', 'Asleep'])
        return centers

    @staticmethod
    def run_kmeans_clustering(dps: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
        kmeans = KMeans(n_clusters=2, init=centers.values, n_init=1, copy_x=True, algorithm="elkan")
        kmeans.fit(dps)
        labels = pd.DataFrame(data=kmeans.labels_, index=dps.index, columns=['Sleep'])
        return labels

    @staticmethod
    def add_back_points_for_which_window_was_to_small(labels: pd.DataFrame, data_with_datetime_index: pd.DataFrame) -> pd.DataFrame:
        all_points = data_with_datetime_index.merge(labels, how='left', left_index=True, right_index=True)['Sleep'].to_frame()
        return all_points

    @staticmethod
    def fill_nas(labels: pd.DataFrame) -> pd.DataFrame:
        labels.interpolate(method='nearest', inplace=True)
        labels.fillna(method='ffill', inplace=True)
        labels.fillna(method='bfill', inplace=True)
        return labels

    @staticmethod
    def reset_index_to_data_index(labels: pd.DataFrame, data_index: pd.Index) -> pd.DataFrame:
        labels.index = data_index
        return labels