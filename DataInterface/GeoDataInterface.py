from typing import List, Union

from DataPreprocessing.GeoData.GeoData import IrisGeoData, TileGeoData, WeatherStationGeoData, CityLatLongGeoData, PollingStationGeoData
from DataPreprocessing.GeoData.GeoMatching import GeoMatchingAPI
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
from DataInterface.DataInterface import DataInterface


class GeoData(DataInterface):
    def __init__(self):
        super().__init__()
        self.geo_data = {gdt.value: None for gdt in GeoDataType}
        self.matching = GeoMatchingAPI.load_matching(load_mappings=True)

    def load(self, geo_data_type: List[GeoDataType] or GeoDataType = None):
        if geo_data_type is None:
            geo_data_type = [gdt for gdt in GeoDataType]
        if not isinstance(geo_data_type, list):
            geo_data_type = [geo_data_type]

        for gdt in geo_data_type:
            self._load_geo_data_type(geo_data_type=gdt)

    def get_geo_data(self, geometry: GeoDataType, subset: List[str] = None, other_geo_data_types: Union[List[GeoDataType], GeoDataType] = None):
        geo_data = self.geo_data[geometry.value].data
        cols_to_keep = [geometry.value, 'geometry']

        if subset is not None:
            geo_data = geo_data[geo_data[geometry.value].isin(subset)]

        if other_geo_data_types is not None:
            if isinstance(other_geo_data_types, GeoDataType):
                other_geo_data_types = [other_geo_data_types]
            other_geo_dt_vals = [gdt.value for gdt in other_geo_data_types]
            geo_data = geo_data.merge(self.matching.data.drop_duplicates(subset=[geometry.value] + other_geo_dt_vals), on=geometry.value, how='left')
            cols_to_keep += [gdt.value for gdt in other_geo_data_types]


        geo_data = geo_data[cols_to_keep].copy()
        return geo_data

    def _load_geo_data_type(self, geo_data_type: GeoDataType):
        if geo_data_type == GeoDataType.IRIS:
            self.geo_data[geo_data_type.IRIS.value] = IrisGeoData()
        elif geo_data_type == GeoDataType.WEATHER_STATION:
            self.geo_data[geo_data_type.WEATHER_STATION.value] = WeatherStationGeoData()
        elif geo_data_type == GeoDataType.CITY:
            self.geo_data[geo_data_type.CITY.value] = CityLatLongGeoData()
        elif geo_data_type == GeoDataType.POLLING_STATION:
            self.geo_data[geo_data_type.POLLING_STATION.value] = PollingStationGeoData()