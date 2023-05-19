from datetime import date, datetime, timedelta
from joblib import Parallel, delayed

import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd

from config import DATA_PATH
from Utils import TrafficType, City, Service


class DataIO:
    @staticmethod
    def load_traffic_data(traffic_type: TrafficType, city: City, service: Service=None, day: date=None) -> xr.DataArray:
        if service is None and day is None:
            return DataIO._city_traffic_data(city=city, traffic_type=traffic_type)
        elif service is None and day is not None:
            return DataIO._city_day_traffic_data(city=city, day=day, traffic_type=traffic_type)
        elif service is not None and day is None:
            return DataIO._city_service_traffic_data(city=city, service=service, traffic_type=traffic_type)
        elif service is not None and day is not None:
            return DataIO._city_service_day_traffic_data(city=city, service=service, day=day, traffic_type=traffic_type)
        else:
            raise ValueError('Invalid parameters')

    @staticmethod
    def _city_traffic_data(city: City, traffic_type: TrafficType) -> xr.DataArray:
        data_vals = []
        days = DataIO.get_days()
        for day in tqdm(days):
            data_vals_day = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(city=city, service=service, day=day, traffic_type=traffic_type) for service in Service)
            data_vals.append(np.stack(data_vals_day, axis=-1))

        data = np.stack(data_vals, axis=-1)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_times(),
                  'service': DataIO.get_services(),
                  'day': DataIO.get_days()}
        dims = ['tile_id', 'time', 'service', 'day']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_service_traffic_data(city: City, service: Service, traffic_type: TrafficType) -> xr.DataArray:
        days = DataIO.get_days()
        data_vals = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(city=city, service=service, day=day, traffic_type=traffic_type) for day in days)
        data = np.stack(data_vals, axis=-1)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_times(),
                  'day': DataIO.get_days()}
        dims = ['tile_id', 'time', 'day']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_day_traffic_data(city: City, day: date, traffic_type: TrafficType) -> xr.DataArray:
        data_vals = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(city=city, service=service, day=day, traffic_type=traffic_type) for service in Service)
        data = np.stack(data_vals, axis=-1)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_times(),
                  'service': DataIO.get_services()}
        dims = ['tile_id', 'time', 'service']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_service_day_traffic_data(city: City, service: Service, day: date, traffic_type: TrafficType) -> xr.DataArray:
        data = DataIO._load_traffic_data_base(city=city, service=service, day=day, traffic_type=traffic_type)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_times()}
        dims = ['tile_id', 'time']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _load_traffic_data_base(city: City, service: Service, day: date, traffic_type: TrafficType) -> pd.DataFrame:
        if traffic_type == TrafficType.DL:
            return DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=False)
        elif traffic_type == TrafficType.UL:
            return DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=True)
        elif traffic_type == TrafficType.B:
            ul_data = DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=True)
            dl_data = DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=False)
            return ul_data + dl_data

    @staticmethod
    def _load_traffic_data_file(city: City, service: Service, day: date, uplink: bool) -> pd.DataFrame:
        file_path = DataIO._get_traffic_data_file_path(city=city, service=service, day=day, uplink=uplink)
        cols = ['tile_id'] + list(DataIO.get_times())
        traffic_data = pd.read_csv(file_path, sep=' ', names=cols)
        traffic_data.set_index('tile_id', inplace=True)
        return traffic_data

    @staticmethod
    def _get_traffic_data_file_path(city: City, service: Service, day: date, uplink: bool) -> str:
        day_str = day.strftime('%Y%m%d')
        ending = 'UL' if uplink else 'DL'
        path = f'{DATA_PATH}/{city.value}/{service.value}/{day_str}/'
        file_name = f'{city.value}_{service.value}_{day_str}_{ending}.txt'
        file_path = path + file_name
        return file_path

    @staticmethod
    def get_times():
        return pd.timedelta_range(start='00:00:00', end='23:59:00', freq='15min')

    @staticmethod
    def get_days():
        return pd.date_range(start='2019-03-16', end='2019-05-31', freq='D')

    @staticmethod
    def get_services():
        return [service.value for service in Service]

    @staticmethod
    def get_tile_ids(city: City):
        data = DataIO._load_traffic_data_file(city=city, service=Service.FACEBOOK_MESSENGER, day=date(2019, 3, 20), uplink=True)
        return data.index
