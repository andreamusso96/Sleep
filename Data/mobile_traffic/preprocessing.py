from datetime import date
from joblib import Parallel, delayed

import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm

from .enums import City, Service, TrafficType, GeoDataType, TrafficDataDimensions, TimeOptions
from . match_iris_tile import geo_matching
from . import config


def city_traffic_data(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City) -> xr.DataArray:
    data_vals = []
    days = TimeOptions.get_days()
    for day in tqdm(days):
        data_vals_day = Parallel(n_jobs=-1)(delayed(load_traffic_data_base)(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day) for service in Service.get_services(traffic_type=traffic_type))
        data_vals.append(np.stack(data_vals_day, axis=-1))

    data = np.stack(data_vals, axis=-1)
    coords = {geo_data_type.value: geo_matching.get_location_list(geo_data_type=geo_data_type, city=city),
              TrafficDataDimensions.TIME.value: TimeOptions.get_times(),
              TrafficDataDimensions.SERVICE.value: Service.get_services(traffic_type=traffic_type, return_values=True),
              TrafficDataDimensions.DAY.value: TimeOptions.get_days()}
    dims = [geo_data_type.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.SERVICE.value, TrafficDataDimensions.DAY.value]
    xar = xr.DataArray(data, coords=coords, dims=dims)
    return xar


def service_traffic_data(traffic_type: TrafficType, geo_data_type: GeoDataType, service: Service) -> xr.DataArray:
    data_vals = []
    location_ids = []
    for city in tqdm(City):
        data_vals_day = Parallel(n_jobs=-1)(delayed(load_traffic_data_base)(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day) for day in TimeOptions.get_days())
        data_vals.append(np.stack(data_vals_day, axis=-1))
        location_ids += geo_matching.get_location_list(geo_data_type=geo_data_type, city=city)

    data = np.concatenate(data_vals, axis=0)
    coords = {geo_data_type.value: location_ids,
              TrafficDataDimensions.TIME.value: TimeOptions.get_times(),
              TrafficDataDimensions.DAY.value: TimeOptions.get_days()}
    dims = [geo_data_type.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.DAY.value]
    xar = xr.DataArray(data, coords=coords, dims=dims)
    return xar


def city_service_traffic_data(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City, service: Service) -> xr.DataArray:
    days = TimeOptions.get_days()
    data_vals = Parallel(n_jobs=-1)(delayed(load_traffic_data_base)(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day) for day in days)
    data = np.stack(data_vals, axis=-1)
    coords = {geo_data_type.value: geo_matching.get_location_list(geo_data_type=geo_data_type, city=city),
              TrafficDataDimensions.TIME.value: TimeOptions.get_times(),
              TrafficDataDimensions.DAY.value: TimeOptions.get_days()}
    dims = [geo_data_type.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.DAY.value]
    xar = xr.DataArray(data, coords=coords, dims=dims)
    return xar


def city_day_traffic_data(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City, day: date) -> xr.DataArray:
    data_vals = Parallel(n_jobs=-1)(delayed(load_traffic_data_base)(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day) for service in Service.get_services(traffic_type=traffic_type))
    data = np.stack(data_vals, axis=-1)
    coords = {geo_data_type.value: geo_matching.get_location_list(geo_data_type=geo_data_type, city=city),
              TrafficDataDimensions.TIME.value: TimeOptions.get_times(),
              TrafficDataDimensions.SERVICE.value: Service.get_services(traffic_type=traffic_type)}
    dims = [geo_data_type.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.SERVICE.value]
    xar = xr.DataArray(data, coords=coords, dims=dims)
    return xar


def city_service_day_traffic_data(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City, service: Service, day: date) -> xr.DataArray:
    data = load_traffic_data_base(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day)
    coords = {geo_data_type.value: geo_matching.get_location_list(city=city, geo_data_type=geo_data_type),
              TrafficDataDimensions.TIME.value: TimeOptions.get_times()}
    dims = [geo_data_type.value, TrafficDataDimensions.TIME.value]
    xar = xr.DataArray(data, coords=coords, dims=dims)
    return xar


def load_traffic_data_base(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City, service: Service, day: date) -> pd.DataFrame:
    if traffic_type == TrafficType.UL_AND_DL or traffic_type == TrafficType.USERS:
        ul_data = load_traffic_data_file(traffic_type=TrafficType.UL, geo_data_type=geo_data_type, city=city, service=service, day=day)
        dl_data = load_traffic_data_file(traffic_type=TrafficType.DL, geo_data_type=geo_data_type, city=city, service=service, day=day)
        traffic = ul_data + dl_data
        if traffic_type == TrafficType.USERS:
            service_data_consumption = Service.get_service_data_consumption(service=service)
            traffic = traffic / service_data_consumption
            return traffic
        else:
            return traffic
    else:
        return load_traffic_data_file(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day)


def load_traffic_data_file(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City, service: Service, day: date) -> pd.DataFrame:
    file_path = config.get_mobile_traffic_data_file_path(traffic_type=traffic_type, geo_data_type=geo_data_type, city=city, service=service, day=day)
    cols = [geo_data_type.value] + list(TimeOptions.get_times())
    traffic_data = pd.read_csv(file_path, sep=' ', names=cols)
    traffic_data.set_index(geo_data_type.value, inplace=True)
    return traffic_data