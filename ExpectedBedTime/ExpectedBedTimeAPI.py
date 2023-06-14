from datetime import time
from typing import List

from DataPreprocessing.TrafficData.Data import CityTrafficData
from ExpectedBedTime.ExpectedBedTime import ExpectedBedTime
from ExpectedBedTime.ExpectedBedTimeCalculator import ExpectedBedTimeCalculator
from ExpectedBedTime.Plots import StartBedTimeRobustness


class ExpectedBedTimeAPI:
    @staticmethod
    def compute_expected_bed_times(traffic_data: CityTrafficData, window_smoothing: int = 3, start_bed_time: List[time] or time = time(21, 45)) -> List[ExpectedBedTime]:
        if isinstance(start_bed_time, time):
            start_bed_time = [start_bed_time]

        expected_bed_times = []
        for sbt in start_bed_time:
            expected_bed_time_calculator = ExpectedBedTimeCalculator(city_traffic_data=traffic_data,
                                                                     start_bed_time=sbt, window_smoothing=window_smoothing)
            expected_bed_time = expected_bed_time_calculator.calculate_expected_bed_time()
            expected_bed_times.append(expected_bed_time)

        return expected_bed_times

    @staticmethod
    def plot_start_bed_time_robustness(traffic_data: CityTrafficData, start_bed_times: List[time] = None):
        if start_bed_times is None:
            start_bed_times = ExpectedBedTimeAPI._start_bed_time_robustness_default_values()

        expected_bed_times = ExpectedBedTimeAPI.compute_expected_bed_times(traffic_data=traffic_data,
                                                                           start_bed_time=start_bed_times)
        start_bed_time_robustness_plots = StartBedTimeRobustness(expected_bed_times=expected_bed_times)
        start_bed_time_robustness_plots.plot_bed_times_for_different_start_bed_time()
        start_bed_time_robustness_plots.plot_iris_quantile_membership_counts_for_different_start_bed_times()

    @staticmethod
    def _start_bed_time_robustness_default_values():
        start_bed_times = [time(21), time(21, 15), time(21, 30), time(21, 45), time(22), time(22, 15), time(22, 30)]
        return start_bed_times


