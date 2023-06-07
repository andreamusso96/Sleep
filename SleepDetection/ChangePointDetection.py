from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import ruptures as rpt

from SleepDetection.Preprocessing import Chunker, Smoother
from Logging.Loggers import SleepInferenceLogger


class ChangePointDetector:
    def __init__(self, traffic_time_series_data):
        self.traffic_time_series_data = traffic_time_series_data
        self.model = 'clinear'
        self.n_change_points = 4
        self.uncertainty_window = 4

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
        change_points = Parallel(n_jobs=-1, verbose=1)(delayed(self._detect_change_points_night_location)(traffic_data_night_location=traffic_data_night[location_id].to_frame(), regressor=regressor) for location_id in traffic_data_night.columns)  # [self._detect_change_points_night_location(traffic_data_night_location=traffic_data_night[location_id].to_frame(), regressor=regressor) for location_id in traffic_data_night.columns]
        change_points = pd.concat(change_points, axis=1)
        change_points.index = pd.MultiIndex.from_product([[traffic_data_night.index[0].date()], change_points.index], names=['date', 'sleep_state'])
        return change_points

    def _detect_change_points_night_location(self, traffic_data_night_location, regressor) -> pd.DataFrame:
        X = np.array([traffic_data_night_location.values.flatten(), regressor]).T
        segmentation, cost = ChangePointDetector._run_change_point_detection_algorithm(X=X, model=self.model, n_change_points=self.n_change_points)
        indices_relevant_change_points, uncertainty_of_relevant_change_points = self._extract_change_points_and_uncertainty_from_segmentation(segmentation=segmentation, cost=cost)
        location_id = traffic_data_night_location.columns[0]
        data_change_points_with_uncertainty = {location_id: traffic_data_night_location.index[indices_relevant_change_points], f'unc_{location_id}': uncertainty_of_relevant_change_points}
        change_points_with_uncertainty = pd.DataFrame(data=data_change_points_with_uncertainty, index=['asleep', 'awake'])
        return change_points_with_uncertainty

    def _extract_change_points_and_uncertainty_from_segmentation(self, segmentation, cost):
        indices_relevant_change_points = [segmentation[1], segmentation[-2]]
        uncertainty_of_relevant_change_points = [ChangePointDetector._estimate_uncertainty_of_change_point(cost=cost, segmentation=segmentation, change_point_index=change_point_index, window=self.uncertainty_window) for change_point_index in indices_relevant_change_points]
        return indices_relevant_change_points, uncertainty_of_relevant_change_points

    @staticmethod
    def _run_change_point_detection_algorithm(X, model, n_change_points):
        algo = rpt.Dynp(model=model, min_size=1, jump=1).fit(X)
        segmentation = np.append([0], algo.predict(n_bkps=n_change_points))
        return segmentation, algo.cost

    @staticmethod
    def _estimate_uncertainty_of_change_point(cost, segmentation, change_point_index, window):
        alternative_change_point_indices = list(range(change_point_index - window, change_point_index + window + 1))
        alternative_change_point_indices.remove(change_point_index)
        alternative_segmentations = [np.where(segmentation==change_point_index, alternative_change_point_index, segmentation) for alternative_change_point_index in alternative_change_point_indices]
        costs_of_alternative_segmentations = np.array([ChangePointDetector._get_cost_of_segmentation(cost=cost, segmentation=alternative_segmentation) for alternative_segmentation in alternative_segmentations])
        cost_of_original_segmentation = ChangePointDetector._get_cost_of_segmentation(cost=cost, segmentation=segmentation)
        relative_costs_of_alternative_segmentations = (costs_of_alternative_segmentations[~np.isnan(costs_of_alternative_segmentations)] - cost_of_original_segmentation) / cost_of_original_segmentation
        uncertainty = np.median(relative_costs_of_alternative_segmentations)
        return uncertainty

    @staticmethod
    def _get_cost_of_segmentation(cost, segmentation):
        diff_segmentation = np.diff(segmentation)
        if np.min(diff_segmentation) < 3:
            SleepInferenceLogger.warning(f'Segmentation {segmentation} has a segment that is too small to compute the cost. Cost is set to nan.')
            return np.nan
        else:
            segmentation_cost = np.sum([cost.error(start=segmentation[i], end=segmentation[i + 1]) for i in range(len(segmentation) - 1)])
            return segmentation_cost



if __name__ == '__main__':
    daily_components = pd.read_csv('Temp/daily_components.csv', index_col=0, parse_dates=True)
    cp_detector = ChangePointDetector(traffic_time_series_data=daily_components)
    cps = cp_detector.detect_change_points()
    cps.to_csv('Temp/change_points.csv')
    a = 0
