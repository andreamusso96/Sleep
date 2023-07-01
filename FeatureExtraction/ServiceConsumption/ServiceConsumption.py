from datetime import time

import pandas as pd


class ServiceConsumption:
    def __init__(self, service_consumption_data: pd.DataFrame, start: time, end: time):
        super().__init__()
        self.data = service_consumption_data
        self.start = start
        self.end = end

    def join(self, other):
        assert self.start == other.start, 'ServiceConsumption can only be joined if they have the same start time'
        assert self.end == other.end, 'ServiceConsumption can only be joined if they have the same end time'
        return ServiceConsumption(service_consumption_data=pd.concat([self.data, other.data], axis=0), start=self.start, end=self.end)