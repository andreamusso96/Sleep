import geopandas as gpd

from Utils import City, AggregationLevel
from config import GEO_DATA_PATH


class GeoData:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.data = self.load()

    def load(self) -> gpd.GeoDataFrame:
        raise NotImplementedError

    def _format(self, data: gpd.GeoDataFrame):
        raise NotImplementedError


class TileGeoData(GeoData):
    def __init__(self, city: City):
        self.city = city
        super().__init__(file_name=f'{GEO_DATA_PATH}/TileGeo/{city.value}.geojson')

    def load(self) -> gpd.GeoDataFrame:
        data = gpd.read_file(filename=self.file_name, dtypes={'tile_id': int})
        data.rename(columns={'tile_id': AggregationLevel.TILE.value}, inplace=True)
        return data


class IrisGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/IrisGeo2019/IRIS_GE.SHP')

    def load(self):
        data = gpd.read_file(filename=self.file_name, dtypes={'CODE_IRIS': str})
        data.to_crs(crs='WGS 84', inplace=True)
        data = data[['CODE_IRIS', 'geometry']]
        data.rename(columns={'CODE_IRIS': AggregationLevel.IRIS.value}, inplace=True)
        return data


if __name__ == '__main__':
    iris = IrisGeoData()