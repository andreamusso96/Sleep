import pandas as pd

from . import config
from . import preprocessing


# Lazy loading
class Data:
    def __init__(self):
        self._data = None

    def load_data(self):
        self._data = preprocessing.load_iris_geo_data()

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load_data()
        return self._data


data = Data()