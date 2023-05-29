import xarray as xr

from SleepDetection.Preprocessing import Preprocessor
from SleepDetection.TimeSeriesDecomposition import Decomposer
from SleepDetection.ChangePointDetection import ChangePointDetector
from SleepDetection.SleepScoreEvaluation import SleepScorer
from Logging.Loggers import SleepInferenceLogger


class DetectionResult:
    def __init__(self, traffic_time_series_data, daily_component_traffic_time_series_data, sleep_change_points, sleep_scores):
        self.traffic_time_series_data = traffic_time_series_data
        self.daily_component_traffic_time_series_data = daily_component_traffic_time_series_data
        self.sleep_change_points = sleep_change_points
        self.sleep_scores = sleep_scores


class Detector:
    def __init__(self, xar_city: xr.DataArray):
        self.xar_city = xar_city

    def detect_sleep_patterns(self):
        SleepInferenceLogger.debug('Sleep detection algorithm started')
        SleepInferenceLogger.debug('Preprocessing data')
        preprocess = Preprocessor(xar_city=self.xar_city)
        traffic_time_series_data = preprocess.preprocess()
        SleepInferenceLogger.debug('Decomposing time series data')
        decomposer = Decomposer(traffic_time_series_data=traffic_time_series_data)
        daily_component_traffic_time_series_data = decomposer.get_daily_component_of_traffic_time_series_data()
        SleepInferenceLogger.debug('Detecting change points')
        change_points_detector = ChangePointDetector(traffic_time_series_data=daily_component_traffic_time_series_data)
        change_points = change_points_detector.detect_change_points()
        SleepInferenceLogger.debug('Computing sleep score')
        sleep_scorer = SleepScorer(sleep_change_points=change_points, traffic_time_series_data=daily_component_traffic_time_series_data)
        sleep_scores = sleep_scorer.compute_sleep_score()
        SleepInferenceLogger.debug('Sleep detection algorithm finished')
        result = DetectionResult(traffic_time_series_data=traffic_time_series_data, daily_component_traffic_time_series_data=daily_component_traffic_time_series_data, sleep_change_points=change_points, sleep_scores=sleep_scores)
        return result



