from typing import List
from enum import Enum

import pandas as pd

from DataPreprocessing.GeoData.GeoData import IrisGeoData, TileGeoData, WeatherStationGeoData, CityLatLongGeoData, PollingStationGeoData
from DataPreprocessing.GeoData.GeoMatching import GeoMatchingAPI
from DataPreprocessing.GeoData.GeoDataType import GeoDataType


class GeoData:
    def __init__(self):
        self.geo_data = {gdt.value: None for gdt in GeoDataType}
        self.matching = GeoMatchingAPI.load_matching(load_mappings=True)

    def load(self, geo_data_type: List[GeoDataType] or GeoDataType = None):
        if geo_data_type is None:
            geo_data_type = [gdt for gdt in GeoDataType]
        if not isinstance(geo_data_type, list):
            geo_data_type = [geo_data_type]

        for gdt in geo_data_type:
            self._load_geo_data_type(geo_data_type=gdt)

    def _load_geo_data_type(self, geo_data_type: GeoDataType):
        if geo_data_type == GeoDataType.IRIS:
            self.geo_data[geo_data_type.IRIS.value] = IrisGeoData()
        elif geo_data_type == GeoDataType.WEATHER_STATION:
            self.geo_data[geo_data_type.WEATHER_STATION.value] = WeatherStationGeoData()
        elif geo_data_type == GeoDataType.CITY:
            self.geo_data[geo_data_type.CITY.value] = CityLatLongGeoData()
        elif geo_data_type == GeoDataType.POLLING_STATION:
            self.geo_data[geo_data_type.POLLING_STATION.value] = PollingStationGeoData()

    def get_matched_geo_data(self, geometry: GeoDataType, other: GeoDataType, codes: List[str] = None):
        geo_data = pd.merge(self.matching.data, self.geo_data[geometry.value].data, left_on=geometry.value, right_on=geometry.value, how='inner')
        geo_data = geo_data[[other.value, geometry.value, 'geometry']]
        geo_data = geo_data.drop_duplicates(subset=[other.value, geometry.value], keep='first')
        if codes is not None:
            geo_data = geo_data[geo_data[other.value].isin(codes)]
        return geo_data.copy()

    def get_areas(self, geo_data_type: GeoDataType):
        return self.geo_data[geo_data_type.value].data.set_index(geo_data_type.value).to_crs(epsg=2154).area


