import pandas as pd

from . import config


# Lazy loading
class _DataLoader:
    def __init__(self):
        self._data = None
        self._metadata = None

    def load_data(self):
        self._data = pd.read_csv(config.get_data_file_path(), index_col=0)

    def load_metadata(self):
        self._metadata = pd.read_csv(config.get_metadata_file_path())

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def metadata(self):
        if self._metadata is None:
            self.load_metadata()
        return self._metadata


_data_loader = _DataLoader()
data = _data_loader.data
metadata = _data_loader.metadata