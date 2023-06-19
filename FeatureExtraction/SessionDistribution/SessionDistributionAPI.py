from datetime import time
from typing import Iterator


from DataInterface.TrafficDataInterface import CityTrafficData
from FeatureExtraction.SessionDistribution.SessionDistribution import SessionDistribution
from FeatureExtraction.SessionDistribution.SessionDistributionCalculator import SessionDistributionCalculator


class SessionDistributionAPI:
    @staticmethod
    def compute_session_distribution(traffic_data: Iterator[CityTrafficData] or CityTrafficData, start: time = time(21, 30), end: time = time(3, 30), window_smoothing: int = 3) -> SleepFeature:
        if isinstance(traffic_data, CityTrafficData):
            traffic_data = [traffic_data]

        session_distribution = None
        for city_traffic_data in traffic_data:
            if session_distribution is None:
                session_distribution = SessionDistributionAPI.compute_session_distribution_city(traffic_data=city_traffic_data, start=start, end=end, window_smoothing=window_smoothing)
            else:
                session_distribution = session_distribution.join(other=SessionDistributionAPI.compute_session_distribution_city(traffic_data=city_traffic_data, start=start, end=end, window_smoothing=window_smoothing))

        return session_distribution

    @staticmethod
    def compute_session_distribution_city(traffic_data: CityTrafficData, start: time, end: time, window_smoothing: int) -> SessionDistribution:
        session_distribution_calculator = SessionDistributionCalculator(city_traffic_data=traffic_data, start=start, end=end, window_smoothing=window_smoothing)
        session_distribution = session_distribution_calculator.calculate_session_distribution()
        return session_distribution