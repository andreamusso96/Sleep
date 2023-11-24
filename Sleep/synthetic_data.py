import mobile_traffic as mt
import numpy as np
import xarray as xr
from tqdm import tqdm
from datetime import time, datetime


def generate_synthetic_dataset(n_services: int = 10, n_times: int = 8):
    data = {}
    services = [s.value for s in mt.Service][:n_services]
    times = [(datetime(2020,1,1) + t).time() for t in mt.TimeOptions.get_times()][:n_times]
    for city in tqdm(mt.City):
        geo_data = mt.geo_tile.get_geo_data(city=city)
        n_tiles = len(geo_data)
        data_city = np.random.randint(low=0, high=100, size=(n_tiles, len(services), len(times)))
        data_city = xr.DataArray(data_city, dims=['tile', 'service', 'time'], coords=[geo_data.index, services, times])
        data[city] = data_city

    return data
