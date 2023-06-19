from datetime import time
from typing import List
from enum import Enum
import webbrowser
import os


import numpy as np
import pandas as pd
import geopandas as gpd

from DataInterface.TrafficDataInterface import Service
from DataInterface.GeoDataInterface import GeoData, GeoDataType


class ServiceConsumptionFeatureName(Enum):
    ENTROPY = 'entropy'
    SIMPSON = 'simpson'


class ServiceConsumptionFeature:
    def __init__(self, service_consumption_by_location: pd.DataFrame):
        self.consumption = service_consumption_by_location

    def _get_service(self, subset: List[Service] = None):
        if subset is None:
            service = list(self.consumption.index)
        else:
            service = [ser.value for ser in subset]
        return service

    def get_consumption_feature(self, feature: ServiceConsumptionFeatureName, subset_location: List[str] = None, subset_service: List[Service] = None):
        service_consumption_by_location = self.consumption.loc[self._get_service(subset=subset_service), subset_location]
        if feature == ServiceConsumptionFeatureName.ENTROPY:
            return self._get_entropy(service_consumption_by_location)
        elif feature == ServiceConsumptionFeatureName.SIMPSON:
            return self._get_simpson(service_consumption_by_location)
        else:
            raise ValueError(f'Feature {feature.value} not supported')

    @staticmethod
    def _get_entropy(consumption):
        probabilities_of_service_usage = consumption / consumption.sum(axis=0)
        entropy = probabilities_of_service_usage.apply(ServiceConsumptionFeature.entropy, axis=1).to_frame(name=ServiceConsumptionFeatureName.ENTROPY.value)
        return entropy

    @staticmethod
    def _get_simpson(consumption):
        probabilities_of_service_usage = consumption / consumption.sum(axis=0)
        simpson = probabilities_of_service_usage.apply(ServiceConsumptionFeature.simpson, axis=1).to_frame(name=ServiceConsumptionFeatureName.SIMPSON.value)
        return simpson

    @staticmethod
    def entropy(x):
        vals = x.values
        return -1 * np.dot(vals, np.log(vals, out=np.zeros_like(vals), where=vals != 0))

    @staticmethod
    def simpson(x):
        vals = x.values
        return 1 - np.dot(vals, vals)






