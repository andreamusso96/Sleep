from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import ruptures as rpt

from SleepDetection.Preprocessing import Chunker, Smoother


class ChangePointDetector:
    def __init__(self, traffic_time_series_data):
        self.traffic_time_series_data = traffic_time_series_data
        self.model = 'clinear'
        self.n_change_points = 4

    def detect_change_points(self):
        smoother = Smoother()
        smoothed_traffic_time_series = smoother.smooth(traffic_time_series_data=self.traffic_time_series_data)
        chunker = Chunker()
        traffic_data_chunked_by_nights = chunker.chunk_series_into_nights(traffic_time_series_data=smoothed_traffic_time_series)
        change_points = [self._detect_change_points_night(traffic_data_night=traffic_data_night) for traffic_data_night in traffic_data_chunked_by_nights]
        change_points = pd.concat(change_points, axis=0)
        return change_points

    def _detect_change_points_night(self, traffic_data_night):
        regressor = np.linspace(0, 1, len(traffic_data_night))
        change_points = Parallel(n_jobs=-1, verbose=1)(delayed(self._detect_change_points_night_location)(traffic_data_night_location=traffic_data_night[location_id].to_frame(), regressor=regressor) for location_id in traffic_data_night.columns)
        change_points = pd.concat(change_points, axis=1)
        change_points.index = pd.MultiIndex.from_product([[traffic_data_night.index[0].date()], change_points.index], names=['date', 'sleep_state'])
        return change_points

    def _detect_change_points_night_location(self, traffic_data_night_location, regressor) -> pd.DataFrame:
        X = np.array([traffic_data_night_location.values.flatten(), regressor]).T
        change_points_indices = ChangePointDetector._run_change_point_detection_algorithm(X=X, model=self.model, n_change_points=self.n_change_points)
        relevant_change_points_indices = [change_points_indices[0], change_points_indices[-2]]
        change_points = pd.DataFrame(data=traffic_data_night_location.index[relevant_change_points_indices], columns=traffic_data_night_location.columns, index=['asleep', 'awake'])
        return change_points

    @staticmethod
    def _run_change_point_detection_algorithm(X, model, n_change_points):
        algo = rpt.Dynp(model=model, min_size=1, jump=1).fit(X)
        change_points_indices = algo.predict(n_bkps=n_change_points)
        return change_points_indices


if __name__ == '__main__':
    daily_components = pd.read_csv('Temp/daily_components.csv', index_col=0, parse_dates=True)
    change_point_detector = ChangePointDetector(traffic_time_series_data=daily_components)
    change_points = change_point_detector.detect_change_points()
    change_points.to_csv('Temp/change_points.csv')
    a = 0
