from typing import List, Dict, Callable

import numpy as np
import xarray as xr
from datetime import time, datetime,timedelta

import mobile_traffic as mt


class MobileData:
    _dataset_path = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/MobileTrafficData'

    def __init__(self, data: Dict[mt.City, xr.DataArray]):
        self.data = data

    @staticmethod
    def _get_dataset_file_path(synthetic: bool, insee_tiles: bool, folder_path: str = None):
        tile_name = 'insee_tile' if insee_tiles else 'netmob_tile'
        if folder_path is None:
            folder = 'synthetic' if synthetic else 'real'
            return f'{MobileData._dataset_path}/{folder}/{tile_name}/mobile_traffic_{{city}}_by_{tile_name}_service_and_time.nc'
        else:
            return f'{folder_path}/mobile_traffic_{{city}}_by_{tile_name}_service_and_time.nc'

    @classmethod
    def load_dataset(cls, synthetic: bool = False, insee_tiles: bool = True, folder_path: str = None):
        data = {}
        file_path = cls._get_dataset_file_path(synthetic=synthetic, insee_tiles=insee_tiles, folder_path=folder_path)
        for c in mt.City:
            data_city = xr.open_dataset(file_path.format(city=c.value.lower())).to_array().squeeze()
            data_city = data_city.assign_coords(time=[datetime.strptime(t, '%H:%M:%S').time() for t in data_city.time.values])
            sorted_times = cls._sort_time_index(time_index=data_city.time.values, reference_time=time(22))
            data_city = data_city.reindex(time=sorted_times)
            data[c] = data_city

        return MobileData(data=data)

    def filter(self, service: List[mt.Service]):
        service_vals = [s.value for s in service]
        data = {c: d.sel(service=service_vals) for c, d in self.data.items()}
        return MobileData(data=data)

    def cities(self):
        return list(self.data.keys())

    def services(self):
        return list(self.data.values())[0].service.values

    def times(self):
        return list(self.data.values())[0].time.values

    def stack_data_along_insee_tile_axis(self) -> xr.DataArray:
        data_stacked = xr.concat([data for city, data in self.data.items()], dim='insee_tile')
        return data_stacked

    @staticmethod
    def _sort_time_index(time_index: List[time], reference_time: time):
        auxiliary_day = datetime(2020, 2, 1)
        auxiliary_dates = []

        for t in time_index:
            if t < reference_time:
                auxiliary_dates.append(datetime.combine(date=auxiliary_day + timedelta(days=1), time=t))
            else:
                auxiliary_dates.append(datetime.combine(date=auxiliary_day, time=t))

        auxiliary_dates.sort()
        sorted_times = [d.time() for d in auxiliary_dates]
        return sorted_times


class TrafficData(MobileData):
    def __init__(self, data: Dict[mt.City, xr.DataArray]):
        super().__init__(data=data)


class ScreenTimeData(MobileData):
    def __init__(self, data: Dict[mt.City, xr.DataArray]):
        super().__init__(data=data)


if __name__ == '__main__':
    time_list = ['00:00:00', '00:15:00', '00:30:00', '00:45:00', '01:00:00', '01:15:00', '01:30:00',
                 '01:45:00', '02:00:00', '02:15:00', '02:30:00', '02:45:00', '03:00:00', '03:15:00',
                 '03:30:00', '03:45:00', '22:00:00', '22:15:00', '22:30:00', '22:45:00', '23:00:00',
                 '23:15:00', '23:30:00']

    time_list = [datetime.strptime(t, '%H:%M:%S').time() for t in time_list]
    MobileData.load_dataset(synthetic=False, insee_tiles=True)