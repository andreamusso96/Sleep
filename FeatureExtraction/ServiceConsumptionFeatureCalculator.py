from typing import List
from enum import Enum

import pandas as pd

from DataInterface.TrafficDataInterface import Service
from FeatureExtraction.FeatureCalculator import FeatureCalculator
from FeatureExtraction.Feature import Feature


class ServiceConsumptionFeatureName(Enum):
    ENTROPY = 'entropy'
    SIMPSON = 'simpson'


class ServiceConsumptionFeatureCalculator(FeatureCalculator):
    def __init__(self, service_consumption_by_location: pd.DataFrame):
        super().__init__()
        self.consumption = service_consumption_by_location

    def _get_service(self, subset: List[Service] = None):
        if subset is None:
            service = list(self.consumption.index)
        else:
            service = [ser.value for ser in subset]
        return service

    def get_consumption_feature(self, feature: ServiceConsumptionFeatureName, subset_location: List[str] = None, subset_service: List[Service] = None) -> Feature:
        service_consumption_by_location = self.consumption[subset_location] if subset_location is not None else self.consumption
        service_consumption_by_location = service_consumption_by_location.loc[self._get_service(subset=subset_service)]

        if feature == ServiceConsumptionFeatureName.ENTROPY:
            probabilities_of_service_usage = service_consumption_by_location / service_consumption_by_location.sum(axis=0)
            entropy_vals = probabilities_of_service_usage.apply(self.entropy, axis=0).to_frame(name=ServiceConsumptionFeatureName.ENTROPY.value)
            entropy = Feature(data=entropy_vals, name=ServiceConsumptionFeatureName.ENTROPY.value)
            return entropy
        elif feature == ServiceConsumptionFeatureName.SIMPSON:
            probabilities_of_service_usage = service_consumption_by_location / service_consumption_by_location.sum(axis=0)
            simpson_vals = probabilities_of_service_usage.apply(self.simpson, axis=0).to_frame(name=ServiceConsumptionFeatureName.SIMPSON.value)
            simpson = Feature(data=simpson_vals, name=ServiceConsumptionFeatureName.SIMPSON.value)
            return simpson
        else:
            raise ValueError(f'Feature {feature.value} not supported')