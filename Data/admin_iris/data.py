import pandas as pd

from . import config


class Data:
    def __init__(self):
        self._data = None
        self._metadata = None

    def load_data(self):
        self._data = pd.read_csv(config.get_data_file_path(), index_col=0, low_memory=False)

    def load_metadata(self):
        self._metadata = pd.read_csv(config.get_metadata_file_path())

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self.load_metadata()
        return self._metadata


data = Data()