from datetime import date, datetime, time

import xarray as xr

from . import preprocessing
from .enums import City, Service, TrafficType, GeoDataType, TrafficDataDimensions
from .city_traffic_data import CityTrafficData
from .utils import day_time_to_datetime_index


def get_city_traffic_data(traffic_type: TrafficType, city: City) -> CityTrafficData:
    data = get_traffic_data(traffic_type=traffic_type, geo_data_type=GeoDataType.IRIS, city=city)
    data = day_time_to_datetime_index(xar=data)
    return CityTrafficData(data=data, city=city, traffic_type=traffic_type, aggregation_level=GeoDataType.IRIS.value)


def get_traffic_data(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City = None, service: Service = None, day: date = None) -> xr.DataArray:
    if service is None and day is None and city is not None:
        return preprocessing.city_traffic_data(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city)
    elif service is None and day is not None and city is not None:
        return preprocessing.city_day_traffic_data(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, day=day)
    elif service is not None and day is None and city is not None:
        return preprocessing.city_service_traffic_data(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service)
    elif service is not None and day is not None and city is not None:
        return preprocessing.city_service_day_traffic_data(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day)
    elif service is not None and day is None and city is None:
        return preprocessing.service_traffic_data(traffic_type=traffic_type, geo_data_type=geo_data_type, service=service)
    else:
        raise ValueError(f'Invalid parameters for DataIO.load_traffic_data: traffic_type={traffic_type}, geo_data_type={geo_data_type}, city={city}, service={service}, day={day}')