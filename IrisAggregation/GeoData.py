import geopandas as gpd

from Utils import City
from config import GEO_DATA_PATH


class GeoData:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.data = self.load()

    def load(self) -> gpd.GeoDataFrame:
        data = gpd.read_file(filename=self.file_name)
        data = self._format(data=data)
        return data

    def _format(self, data: gpd.GeoDataFrame):
        raise NotImplementedError


class TileGeoData(GeoData):
    def __init__(self, city: City):
        self.city = city
        super().__init__(file_name=f'{GEO_DATA_PATH}/TileGeo/{city.value}.geojson')

    def _format(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return data


class IrisGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/IrisGeo2019/IRIS_GE.SHP')

    def _format(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        data.to_crs(crs='WGS 84', inplace=True)
        data = data[['CODE_IRIS', 'geometry']]
        return data