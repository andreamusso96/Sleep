import pandas as pd

from . import config


class Data:
    def __init__(self):
        self._data = None

    def load_data(self):
        self._data = pd.read_csv(config.get_data_file_path(), dtype={'polling_station': str})
        self._data.set_index('polling_station', inplace=True)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load_data()
        return self._data


data = Data()