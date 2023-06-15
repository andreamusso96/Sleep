import geopandas as gpd
import pandas as pd
from unidecode import unidecode

from Utils import City
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
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
        data.rename(columns={'tile_id': GeoDataType.TILE.value}, inplace=True)
        return data


class IrisGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/IrisGeo2019/IRIS_GE.SHP')

    def load(self):
        data = gpd.read_file(filename=self.file_name, dtypes={'CODE_IRIS': str})
        data.to_crs(crs='WGS 84', inplace=True)
        data = data[['CODE_IRIS', 'geometry']].copy()
        data.rename(columns={'CODE_IRIS': GeoDataType.IRIS.value}, inplace=True)
        return data


class CityLatLongGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/CityGeo/city_lat_long.csv')

    def load(self):
        data = pd.read_csv(self.file_name, sep=',', header=0)
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(x=data['lng'], y=data['lat']))
        gdf = gdf.rename(columns={'city': GeoDataType.CITY.value})
        gdf = gdf[[GeoDataType.CITY.value, 'geometry']].copy()
        gdf[GeoDataType.CITY.value] = gdf[GeoDataType.CITY.value].apply(lambda city_name: CityLatLongGeoData._normalize_city_names(name=city_name))
        gdf.set_crs(crs='WGS 84', inplace=True)
        return gdf

    @staticmethod
    def _normalize_city_names(name: str):
        normalized_name = unidecode(name)
        if normalized_name.startswith('Le '):
            normalized_name = normalized_name.split(' ')[1]
        return normalized_name


class WeatherStationGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/WeatherStationGeo/weather_station_locations.json')

    def load(self):
        data = gpd.read_file(self.file_name)
        return data


class PollingStationGeoData(GeoData):
    def __init__(self):
        super().__init__(f'{GEO_DATA_PATH}/PollingStationGeo/bureaux-vote-france-2017.geojson')

    def load(self) -> gpd.GeoDataFrame:
        data = gpd.read_file(self.file_name)
        data.dropna(subset=['geometry'], inplace=True)
        column_names = {
            'code_insee': 'municipality_code',
            'code_bureau_vote': 'polling_station_code',
            'geometry': 'geometry'
        }
        data = data[column_names.keys()].copy()
        data.rename(columns=column_names, inplace=True)
        data = PollingStationGeoData._format_polling_station_codes(data=data)
        data[GeoDataType.POLLING_STATION.value] = data['municipality_code'] + data['polling_station_code']
        data.dropna(subset=['geometry'], inplace=True)
        data.sort_values(by=['municipality_code', 'polling_station_code'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    @staticmethod
    def _format_polling_station_codes(data):
        data['polling_station_code_is_correct'] = data['polling_station_code'].apply(lambda x: PollingStationGeoData.is_castable_to_int(x))
        data = data.loc[data['polling_station_code_is_correct']].copy()
        data.drop(columns=['polling_station_code_is_correct'], inplace=True)
        data['polling_station_code'] = data['polling_station_code'].apply(lambda x: str(x).zfill(4))
        return data

    @staticmethod
    def is_castable_to_int(value):
        try:
            int(value)
            return True
        except ValueError:
            return False


if __name__ == '__main__':
    PollingStationGeoData()