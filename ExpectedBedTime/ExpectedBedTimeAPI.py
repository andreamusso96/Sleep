from datetime import time
from typing import List

from DataPreprocessing.TrafficData.Data import CityTrafficData
from ExpectedBedTime.ExpectedBedTime import ExpectedBedTime
from ExpectedBedTime.ExpectedBedTimeCalculator import ExpectedBedTimeCalculator


class ExpectedBedTimeAPI:
    @staticmethod
    def compute_expected_bed_times(city_traffic_data: CityTrafficData, window_smoothing: int = 3, start_bed_time: List[time] or time = time(21, 45)) -> List[ExpectedBedTime]:
        if isinstance(start_bed_time, time):
            start_bed_time = [start_bed_time]

        expected_bed_times = []
        for sbt in start_bed_time:
            expected_bed_time_calculator = ExpectedBedTimeCalculator(city_traffic_data=city_traffic_data,
                                                                     start_bed_time=sbt, window_smoothing=window_smoothing)
            expected_bed_time = expected_bed_time_calculator.calculate_expected_bed_time()
            expected_bed_times.append(expected_bed_time)

        return expected_bed_times