import sys

import mobile_traffic as mt
import xarray as xr
from datetime import datetime

from aggregate_geo import aggregate_netmob_tile_level_traffic_data_city_to_insee_tile_level


def aggregate_netmob_tile_to_insee_tile_level_and_save(folder_path: str, city: mt.City):
    file_path = f'{folder_path}/mobile_traffic_{city.value.lower()}_by_tile_service_and_time.nc'
    data_city = load_netmob_tile_dataset_city(file_path=file_path)
    agg = aggregate_netmob_tile_level_traffic_data_city_to_insee_tile_level(traffic_data_netmob_tile=data_city, city=city)
    time_as_str = [str(t) for t in agg.time.values]
    agg = agg.assign_coords(time=time_as_str)
    agg.to_netcdf(f'{folder_path}/mobile_traffic_{city.value.lower()}_by_insee_tile_service_and_time.nc')


def load_netmob_tile_dataset_city(file_path: str):
    data_city = xr.open_dataset(file_path).to_array().squeeze()
    data_city = data_city.assign_coords(time=[datetime.strptime(t, '%H:%M:%S').time() for t in data_city.time.values])
    return data_city


if __name__ == '__main__':
    fp = '/cluster/work/gess/coss/users/anmusso/Temp'
    c = mt.City(sys.argv[1])
    aggregate_netmob_tile_to_insee_tile_level_and_save(folder_path=fp, city=c)