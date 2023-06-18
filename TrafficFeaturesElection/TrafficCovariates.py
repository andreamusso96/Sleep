from datetime import time
from typing import List
import numpy as np
import pandas as pd

from DataPreprocessing.TrafficData.Data import CityTrafficData
from Utils import Service


class TrafficFeaturesCalculator:
    def __init__(self, city_traffic_data: CityTrafficData, start_time: time = time(18), end_time: time = time(23)):
        self.city_traffic_data = city_traffic_data
        self.start_time = start_time
        self.end_time = end_time
        self.service_consumption_by_location = self.city_traffic_data.get_service_consumption_by_location(start=start_time, end=end_time)

    def get_consumption_diversity(self, services: List[Service] = None):
        if services is None:
            services_ = list(self.service_consumption_by_location.index)
        else:
            services_ = [ser.value for ser in services]

        service_consumption_by_location = self.service_consumption_by_location.loc[services_]
        probabilities_of_service_usage = service_consumption_by_location / service_consumption_by_location.sum(axis=0)
        simpson_index = 1 - np.square(probabilities_of_service_usage).sum(axis=0)
        entropy = -1 * (probabilities_of_service_usage * np.log(probabilities_of_service_usage)).sum(axis=0)
        diversity = pd.concat([simpson_index, entropy], axis=1)
        diversity.columns = ['simpson', 'entropy']
        return diversity




