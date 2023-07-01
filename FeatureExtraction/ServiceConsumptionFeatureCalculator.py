from typing import List
from enum import Enum

import numpy as np
import pandas as pd

from DataInterface.TrafficDataInterface import Service
from DataInterface.AdminDataInterface import AdminData
from FeatureExtraction.FeatureCalculator import FeatureCalculator
from FeatureExtraction.Feature import Feature
from FeatureExtraction.ServiceConsumption.ServiceConsumption import ServiceConsumption


class ServiceConsumptionFeatureName(Enum):
    ENTROPY = 'entropy'
    SIMPSON = 'simpson'
    PRO_CAPITA = 'pro_capita'
    SHARES = 'shares'


class ServiceConsumptionFeatureCalculator(FeatureCalculator):
    def __init__(self, service_consumption: ServiceConsumption,  admin_data: AdminData):
        super().__init__()
        self.consumption = service_consumption
        self.admin_data = admin_data

    def _get_service(self, subset: List[Service] = None):
        if subset is None:
            service = list(self.consumption.data.columns)
        else:
            service = [ser.value for ser in subset]
        return service

    def get_consumption_feature(self, feature: ServiceConsumptionFeatureName, subset_location: List[str] = None, subset_service: List[Service] = None) -> Feature:
        service_consumption_by_location = self.consumption.data.loc[subset_location] if subset_location is not None else self.consumption

        if feature == ServiceConsumptionFeatureName.ENTROPY:
            service_consumption_by_location = service_consumption_by_location[self._get_service(subset=subset_service)]
            return self._get_consumption_entropy(service_consumption_by_location=service_consumption_by_location)
        elif feature == ServiceConsumptionFeatureName.SIMPSON:
            service_consumption_by_location = service_consumption_by_location[self._get_service(subset=subset_service)]
            return self._get_consumption_simpson(service_consumption_by_location=service_consumption_by_location)
        elif feature == ServiceConsumptionFeatureName.PRO_CAPITA:
            service_consumption_by_location = service_consumption_by_location[self._get_service(subset=subset_service)]
            return self._get_consumption_pro_capita(service_consumption_by_location=service_consumption_by_location)
        elif feature == ServiceConsumptionFeatureName.SHARES:
            return self._get_consumption_shares_services(service_consumption_by_location=service_consumption_by_location, services=self._get_service(subset=subset_service))
        else:
            raise ValueError(f'Feature {feature.value} not supported')

    def _get_consumption_entropy(self, service_consumption_by_location: pd.DataFrame) -> Feature:
        probabilities_of_service_usage = self._get_probabilities_of_service_usage(service_consumption_by_location=service_consumption_by_location)
        entropy_vals = probabilities_of_service_usage.apply(self.entropy, axis=1).to_frame(name=ServiceConsumptionFeatureName.ENTROPY.value)
        entropy = Feature(data=entropy_vals, name=ServiceConsumptionFeatureName.ENTROPY.value)
        return entropy

    def _get_consumption_simpson(self, service_consumption_by_location: pd.DataFrame) -> Feature:
        probabilities_of_service_usage = self._get_probabilities_of_service_usage(service_consumption_by_location=service_consumption_by_location)
        simpson_vals = probabilities_of_service_usage.apply(self.simpson, axis=1).to_frame(name=ServiceConsumptionFeatureName.SIMPSON.value)
        simpson = Feature(data=simpson_vals, name=ServiceConsumptionFeatureName.SIMPSON.value)
        return simpson

    def _get_consumption_pro_capita(self, service_consumption_by_location: pd.DataFrame) -> Feature:
        total_consumption_location = service_consumption_by_location.sum(axis=1).to_frame(name='total_consumption')
        population = self.admin_data.data['P19_POP'].to_frame()
        consumption_and_population = pd.merge(total_consumption_location, population, left_index=True, right_index=True)
        consumption_pro_capita = np.divide(consumption_and_population['total_consumption'], consumption_and_population['P19_POP'], out=np.zeros_like(consumption_and_population['total_consumption']), where=consumption_and_population['P19_POP'] != 0).to_frame(name=ServiceConsumptionFeatureName.PRO_CAPITA.value)
        pro_capita = Feature(data=consumption_pro_capita, name=ServiceConsumptionFeatureName.PRO_CAPITA.value)
        return pro_capita

    def _get_consumption_shares_services(self, service_consumption_by_location: pd.DataFrame, services: List[str]) -> Feature:
        consumption_shares_all_services = self._get_probabilities_of_service_usage(service_consumption_by_location=service_consumption_by_location)
        consumption_shares_services = consumption_shares_all_services[services].sum(axis=1).to_frame(name=ServiceConsumptionFeatureName.SHARES.value)
        consumption_shares_services = Feature(data=consumption_shares_services, name=ServiceConsumptionFeatureName.SHARES.value)
        return consumption_shares_services

    @staticmethod
    def _get_probabilities_of_service_usage(service_consumption_by_location: pd.DataFrame) -> pd.DataFrame:
        probabilities_of_service_usage = (service_consumption_by_location.T / service_consumption_by_location.sum(axis=1)).T
        return probabilities_of_service_usage