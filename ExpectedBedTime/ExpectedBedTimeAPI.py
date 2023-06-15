from datetime import time
from typing import List, Iterator


from DataPreprocessing.TrafficData.Data import CityTrafficData
from ExpectedBedTime.ExpectedBedTime import ExpectedBedTime
from ExpectedBedTime.ExpectedBedTimeCalculator import ExpectedBedTimeCalculator
from ExpectedBedTime.Plots import StartBedTimeRobustness
from Utils import City


class ExpectedBedTimeAPI:
    @staticmethod
    def compute_expected_bed_time(traffic_data: Iterator[CityTrafficData] or CityTrafficData, window_smoothing: int = 3, start_bed_time: time = time(21, 45)) -> ExpectedBedTime:
        if isinstance(traffic_data, CityTrafficData):
            traffic_data = [traffic_data]

        expected_bed_time = None
        for city_traffic_data in traffic_data:
            if expected_bed_time is None:
                expected_bed_time = ExpectedBedTimeAPI._compute_expected_bed_time_city(traffic_data=city_traffic_data,window_smoothing=window_smoothing, start_bed_time=start_bed_time)
            else:
                expected_bed_time = expected_bed_time.join(ExpectedBedTimeAPI._compute_expected_bed_time_city(traffic_data=city_traffic_data,window_smoothing=window_smoothing, start_bed_time=start_bed_time))

        return expected_bed_time

    @staticmethod
    def _compute_expected_bed_time_city(traffic_data: CityTrafficData, window_smoothing: int, start_bed_time: time) -> ExpectedBedTime:
        expected_bed_time_calculator = ExpectedBedTimeCalculator(city_traffic_data=traffic_data,
                                                                 start_bed_time=start_bed_time, window_smoothing=window_smoothing)
        expected_bed_time = expected_bed_time_calculator.calculate_expected_bed_time()
        return expected_bed_time

    @staticmethod
    def plot_start_bed_time_robustness(traffic_data: Iterator[CityTrafficData] or CityTrafficData, start_bed_times: List[time] = None):
        if start_bed_times is None:
            start_bed_times = ExpectedBedTimeAPI._start_bed_time_robustness_default_values()

        expected_bed_times = [ExpectedBedTimeAPI.compute_expected_bed_time(traffic_data=traffic_data, start_bed_time=start_bed_time) for start_bed_time in start_bed_times]
        start_bed_time_robustness_plots = StartBedTimeRobustness(expected_bed_times=expected_bed_times)
        start_bed_time_robustness_plots.plot_bed_times_for_different_start_bed_time()
        start_bed_time_robustness_plots.plot_iris_quantile_membership_counts_for_different_start_bed_times()

    @staticmethod
    def _start_bed_time_robustness_default_values():
        start_bed_times = [time(21), time(21, 15), time(21, 30), time(21, 45), time(22), time(22, 15), time(22, 30)]
        return start_bed_times


