from typing import List

import pandas as pd

from . import config


# Lazy loading
class Data:
    def __init__(self):
        self._data = None
        self._matching_iris_polling_station = None

    def load_data(self):
        self._data = pd.read_csv(config.get_data_file_path(), dtype={'polling_station': str}, low_memory=False)

    def load_matching_iris_polling_station(self):
        self._matching_iris_polling_station = pd.read_csv(config.get_matching_iris_polling_station_file_path(), dtype={'polling_station': str, 'iris': str, 'radius': float}, low_memory=False)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def matching_iris_polling_station(self) -> pd.DataFrame:
        if self._matching_iris_polling_station is None:
            self.load_matching_iris_polling_station()
        return self._matching_iris_polling_station




data = Data()