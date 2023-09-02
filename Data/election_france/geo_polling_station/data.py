import geopandas as gpd

from . import config


class Data:
    def __init__(self):
        self._data = None

    def load_data(self):
        self._data = gpd.read_file(config.get_data_file_path(), dtype={'polling_station': str})
        self._data.set_index('polling_station', inplace=True)

    @property
    def data(self) -> gpd.GeoDataFrame:
        if self._data is None:
            self.load_data()
        return self._data


data = Data()