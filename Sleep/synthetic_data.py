import mobile_traffic as mt
import numpy as np
import xarray as xr
from tqdm import tqdm
from datetime import time, datetime


def generate_synthetic_dataset():
    data = {}
    services = [s.value for s in mt.Service][:10]
    times = [(datetime(2020,1,1) + t).time() for t in mt.TimeOptions.get_times()][:8]
    for city in tqdm(mt.City):
        geo_data = mt.geo_tile.get_geo_data(city=city)
        n_tiles = len(geo_data)
        data_city = np.random.randint(low=0, high=100, size=(n_tiles, len(services), len(times)))
        data_city = xr.DataArray(data_city, dims=['tile', 'service', 'time'], coords=[geo_data.index, services, times])
        data[city] = data_city

    return data


def load_synthetic_dataset(folder_path: str, insee_tiles: bool = False):
    data = {}
    tile_name = 'insee_tile' if insee_tiles else 'tile'
    for c in mt.City:
        data_city = xr.open_dataset(f'{folder_path}/mobile_traffic_{c.value.lower()}_by_{tile_name}_service_and_time.nc').to_array().squeeze()
        data_city = data_city.assign_coords(time=[datetime.strptime(t, '%H:%M:%S').time() for t in data_city.time.values])
        data[c] = data_city
    return data


def save_synthetic_dataset(folder_path: str):
    d = generate_synthetic_dataset()
    for c, dat in d.items():
        time_as_str = [str(t) for t in dat.time.values]
        data_ = dat.assign_coords(time=time_as_str)
        data_.to_netcdf(f'{folder_path}/mobile_traffic_{c.value.lower()}_by_tile_service_and_time.nc')


if __name__ == '__main__':
    fp = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/SyntheticData'
    save_synthetic_dataset(folder_path=fp)
    load_synthetic_dataset(folder_path=fp)

