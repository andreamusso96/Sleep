from enum import Enum


class GeoDataType(Enum):
    TILE = 'tile'
    IRIS = 'iris'
    WEATHER_STATION = 'weather_station'
    CITY = 'city'
    POLLING_STATION = 'polling_station'