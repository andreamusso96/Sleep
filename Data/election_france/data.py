import pandas as pd
import Data.geo_france as gf

from . import config


# Lazy loading
class _DataLoader:
    def __init__(self):
        self._data = None

    def load_data(self):
        self._data = pd.read_csv(config.get_data_file())
        self._data.set_index(gf.GeoDataType.POLLING_STATION.value, inplace=True)

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data


_data_loader = _DataLoader()
data = _data_loader.data