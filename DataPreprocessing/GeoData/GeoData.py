import geopandas as gpd
import pandas as pd
from unidecode import unidecode

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
        data = data[['CODE_IRIS', 'geometry']].copy()
        data.rename(columns={'CODE_IRIS': AggregationLevel.IRIS.value}, inplace=True)
        return data


class CityLatLongGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/city_lat_long.csv')

    def load(self):
        data = pd.read_csv(self.file_name, sep=',', header=0)
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(x=data['lng'], y=data['lat']))
        gdf = gdf[['city', 'geometry']]
        gdf['city'] = gdf['city'].apply(lambda city_name: CityLatLongGeoData._normalize_city_names(name=city_name))
        gdf.set_crs(crs='WGS 84', inplace=True)
        return gdf

    @staticmethod
    def _normalize_city_names(name: str):
        normalized_name = unidecode(name)
        if normalized_name.startswith('Le '):
            normalized_name = normalized_name.split(' ')[1]
        return normalized_name


if __name__ == '__main__':
    c = CityLatLongGeoData()
    c.load()