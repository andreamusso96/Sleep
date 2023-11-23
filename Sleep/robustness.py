from typing import Dict, List, Callable

import xarray as xr
import pandas as pd
import numpy as np

import mobile_traffic as mt
import engineer_features as ef
import transform


def screen_time_data_sample(traffic_data: Dict[mt.City, xr.DataArray], traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]], n_samples: int) -> Dict[mt.City, xr.DataArray]:
    screen_time_data = {city: city_screen_time_data_sample(traffic_data=traffic_data[city], traffic_per_minute_sampler=traffic_per_minute_sampler, n_samples=n_samples) for city in traffic_data}
    return screen_time_data


def city_screen_time_data_sample(traffic_data: xr.DataArray, traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]], n_samples: int) -> xr.DataArray:
    screen_time_city = []
    for i, service in enumerate(traffic_per_minute_sampler):
        traffic_data_service = traffic_data.sel(service=service.value)

        cum_sum_traffic_per_minute_sample = _get_cum_sum_of_traffic_per_minute_samples(max_traffic_data=float(traffic_data_service.max()), traffic_per_minute_sampler=traffic_per_minute_sampler[service], n_samples=n_samples)

        screen_time_service_minutes = xr.apply_ufunc(lambda x: np.searchsorted(cum_sum_traffic_per_minute_sample, x), traffic_data_service)
        screen_time_city.append(screen_time_service_minutes)

    screen_time_city = xr.concat(screen_time_city, dim='service')
    return screen_time_city


def _get_cum_sum_of_traffic_per_minute_samples(max_traffic_data: float, traffic_per_minute_sampler: Callable[[int], np.ndarray], n_samples: int) -> np.ndarray:
    traffic_consumption_sample, sum_traffic_consumption_sample = [], 0
    while sum_traffic_consumption_sample < 1.2*max_traffic_data:
        traffic_consumption_sample += list(traffic_per_minute_sampler(n_samples))
        sum_traffic_consumption_sample = np.sum(traffic_consumption_sample)
    cum_sum_traffic_consumption_sample = np.cumsum(traffic_consumption_sample)
    return cum_sum_traffic_consumption_sample


def get_normal_distribution_sampler(mean: float, std: float) -> Callable[[int], np.ndarray]:
    def normal_distribution_sampler(n_samples: int) -> np.ndarray:
        return np.random.normal(loc=mean, scale=std, size=n_samples)

    return normal_distribution_sampler



if __name__ == '__main__':
    from synthetic_data import load_synthetic_dataset
    from tqdm import tqdm
    td = load_synthetic_dataset(folder_path='/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/SyntheticData', insee_tiles=True)
    income_quantiles = [0.3, 0.7]
    serv = [mt.Service.TWITCH, mt.Service.ORANGE_TV, mt.Service.MICROSOFT_AZURE, mt.Service.APPLE_ICLOUD, mt.Service.WEB_GAMES, mt.Service.PLAYSTATION, mt.Service.TEAMVIEWER, mt.Service.WEB_WEATHER, mt.Service.GOOGLE_MEET, mt.Service.TWITTER]
    m = [800, 900, 300, 300, 100, 900, 900, 5, 900, 100]
    s = [100, 100, 100, 100, 100, 100, 100, 1, 100, 100]
    samplers = {ser: get_normal_distribution_sampler(mean=m[i]/100, std=s[i]/100) for i, ser in enumerate(serv)}

    td_samples = [screen_time_data_sample(traffic_data=td, traffic_per_minute_sampler=samplers, n_samples=100) for _ in tqdm(range(100))]