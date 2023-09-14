from typing import Dict

import pandas as pd
import geopandas as gpd

from . import preprocessing


# Lazy loading
class Data:
    def __init__(self):
        self._data = None

    def load_data(self):
        self._data = preprocessing.load_tile_geo_data()

    @property
    def data(self) -> Dict[int, gpd.GeoDataFrame]:
        if self._data is None:
            self.load_data()
        return self._data


data = Data()