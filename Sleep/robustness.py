from typing import Dict, List, Callable, Any, Union, Tuple

import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import mobile_traffic as mt

import engineer_features as ef
import transform
from mobile_data import MobileData, TrafficData, ScreenTimeData


def rca_income_and_service_samples__screen_time(traffic_data: TrafficData, income_quantiles: List[float], n_samples: int, traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]]) -> xr.DataArray:
    screen_time_data_samples = get_screen_time_data_samples(traffic_data=traffic_data, n_samples=n_samples, traffic_per_minute_sampler=traffic_per_minute_sampler)
    rca_samples = _rca_income_and_service_samples(mobile_data_samples=screen_time_data_samples, income_quantiles=income_quantiles)
    return rca_samples


def rca_income_and_service_samples__amenity(mobile_data: MobileData, thresholds_and_buffers: List[Tuple[int, float]], income_quantiles: List[float]) -> xr.DataArray:
    amenity_data_samples = get_amenity_data_samples(mobile_data=mobile_data, thresholds_and_buffers=thresholds_and_buffers)
    rca_samples = _rca_income_and_service_samples(mobile_data_samples=amenity_data_samples, income_quantiles=income_quantiles)
    return rca_samples


def night_screen_index_samples__screen_time(traffic_data: TrafficData, n_samples: int, traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]]) -> pd.DataFrame:
    screen_time_data_samples = get_screen_time_data_samples(traffic_data=traffic_data, n_samples=n_samples, traffic_per_minute_sampler=traffic_per_minute_sampler)
    nsi_samples = _night_screen_index_samples(screen_time_data_samples=screen_time_data_samples)  # noqa
    return nsi_samples


def night_screen_index_samples__amenity(screen_time_data: ScreenTimeData, thresholds_and_buffers: List[Tuple[int, float]]) -> pd.DataFrame:
    amenity_data_samples = get_amenity_data_samples(mobile_data=screen_time_data, thresholds_and_buffers=thresholds_and_buffers)
    nsi_samples = _night_screen_index_samples(screen_time_data_samples=amenity_data_samples)  # noqa
    return nsi_samples


def get_amenity_data_samples(mobile_data: MobileData, thresholds_and_buffers: List[Tuple[int, float]]) -> Dict[str, MobileData]:
    amenity_data_samples = {str((threshold, buffer_size_m)): filter_out_tiles_with_many_amenities_open_at_night(mobile_data=mobile_data, buffer_size_m=buffer_size_m, threshold=threshold) for (threshold, buffer_size_m) in thresholds_and_buffers}
    return amenity_data_samples


def get_screen_time_data_samples(traffic_data: TrafficData, n_samples: int, traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]]) -> Dict[str, MobileData]:
    mobile_data_ = traffic_data.filter(service=list(traffic_per_minute_sampler.keys()))
    screen_time_data_samples = {str(i): screen_time_data_sample(traffic_data=mobile_data_, traffic_per_minute_sampler=traffic_per_minute_sampler) for i in range(n_samples)}
    return screen_time_data_samples


def _rca_income_and_service_samples(mobile_data_samples: Dict[str, MobileData], income_quantiles: List[float]) -> xr.DataArray:
    rca = [transform.rca_income_and_services(mobile_data=mobile_data_samples[sample_name], income_quantiles=income_quantiles) for sample_name in mobile_data_samples]
    data = np.stack([r.values for r in rca], axis=-1)
    dims = ['income_category', 'service', 'sample']
    coords = [rca[0].index, rca[0].columns, list(mobile_data_samples.keys())]
    rca = xr.DataArray(data, dims=dims, coords=coords)
    return rca


def _night_screen_index_samples(screen_time_data_samples: Dict[str, ScreenTimeData]) -> pd.DataFrame:
    night_screen_indices = []
    for sample_name in screen_time_data_samples:
        night_screen_index_sample = ef.night_screen_index_insee_tile(screen_time_data=screen_time_data_samples[sample_name])
        night_screen_index_sample.rename(columns={'night_screen_index': f'night_screen_index_{sample_name}'}, inplace=True)
        night_screen_index_sample.drop(columns=['city'], inplace=True)
        night_screen_indices.append(night_screen_index_sample)

    night_screen_indices = pd.concat(night_screen_indices, axis=1)
    return night_screen_indices


# Amenity data sampling


def filter_out_tiles_with_many_amenities_open_at_night(mobile_data: MobileData, threshold: int, buffer_size_m: float) -> MobileData:
    data = {}
    for city in mobile_data.cities():
        mobile_data_city = mobile_data.data[city]
        insee_tiles_city = mobile_data_city.insee_tile.values
        amenities_counts = ef.log2_amenity_counts_insee_tile(insee_tiles=insee_tiles_city, buffer_size_m=buffer_size_m, amenity_type=ef.AmenityType.OPEN_AT_NIGHT)
        insee_tiles_with_many_amenities_open_at_night = amenities_counts[amenities_counts['log2_oan_amenity_counts'] > threshold].index
        insee_tiles_to_keep = np.setdiff1d(insee_tiles_city, insee_tiles_with_many_amenities_open_at_night)
        mobile_data_city = mobile_data_city.sel(insee_tile=insee_tiles_to_keep)
        data[city] = mobile_data_city

    return MobileData(data=data)


# Screen time robustness data sampling
def screen_time_data_sample(traffic_data: TrafficData, traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]]) -> ScreenTimeData:
    screen_time_data = {city: city_screen_time_data_sample(traffic_data=traffic_data.data[city], traffic_per_minute_sampler=traffic_per_minute_sampler) for city in traffic_data.cities()}
    screen_time_data = ScreenTimeData(data=screen_time_data)
    return screen_time_data


def city_screen_time_data_sample(traffic_data: xr.DataArray, traffic_per_minute_sampler: Dict[mt.Service, Callable[[int], np.ndarray]]) -> xr.DataArray:
    screen_time_city = []
    for service in tqdm(traffic_per_minute_sampler):
        traffic_data_service = traffic_data.sel(service=service.value)

        cum_sum_traffic_per_minute_sample = _get_cum_sum_of_traffic_per_minute_samples(max_traffic_data=float(traffic_data_service.max()), traffic_per_minute_sampler=traffic_per_minute_sampler[service])

        screen_time_service_minutes = xr.apply_ufunc(lambda x: np.searchsorted(cum_sum_traffic_per_minute_sample, x), traffic_data_service)
        screen_time_city.append(screen_time_service_minutes)

    data = np.stack([s.values for s in screen_time_city], axis=1)
    dims = ['insee_tile', 'service', 'time']
    coords = [traffic_data.insee_tile.values, list(traffic_per_minute_sampler.keys()), traffic_data.time.values]
    screen_time_city = xr.DataArray(data, dims=dims, coords=coords)
    return screen_time_city


def _get_cum_sum_of_traffic_per_minute_samples(max_traffic_data: float, traffic_per_minute_sampler: Callable[[int], np.ndarray]) -> np.ndarray:
    traffic_generation_sample, sum_traffic_generation_sample = [], 0
    n_samples = _heuristic_for_number_of_samples(max_traffic_data=max_traffic_data, traffic_per_minute_sampler=traffic_per_minute_sampler)
    while sum_traffic_generation_sample < 1.2 * max_traffic_data:
        traffic_generation_sample += list(traffic_per_minute_sampler(n_samples))
        sum_traffic_generation_sample = np.sum(traffic_generation_sample)
    cum_sum_traffic_consumption_sample = np.cumsum(traffic_generation_sample)
    return cum_sum_traffic_consumption_sample


def _heuristic_for_number_of_samples(max_traffic_data: float, traffic_per_minute_sampler: Callable[[int], np.ndarray]) -> int:
    traffic_generation_samples = traffic_per_minute_sampler(10 ** 4)
    mean_traffic_generation_sample = np.mean(traffic_generation_samples)
    n_samples = int(0.5 * max_traffic_data / mean_traffic_generation_sample)
    return n_samples


def get_normal_distribution_sampler(mean: float, std: float) -> Callable[[int], np.ndarray]:
    def normal_distribution_sampler(n_samples: int) -> np.ndarray:
        sample = np.random.normal(loc=mean, scale=std, size=n_samples)
        sample = np.maximum(sample, 0.001)
        return sample

    return normal_distribution_sampler


def service_traffic_per_minute_sampler() -> Dict[mt.Service, Callable[[int], np.ndarray]]:
    hourly_traffic_mb = {
        mt.Service.TWITCH: 800,
        mt.Service.ORANGE_TV: 900,
        mt.Service.WEB_GAMES: 100,
        mt.Service.WEB_WEATHER: 5,
        mt.Service.TWITTER: 100,
        mt.Service.APPLE_MUSIC: 150,
        mt.Service.WEB_ADS: 30,
        mt.Service.SOUNDCLOUD: 150,
        mt.Service.WIKIPEDIA: 15,
        mt.Service.WEB_FOOD: 50,
        mt.Service.YOUTUBE: 800,
        mt.Service.PINTEREST: 200,
        mt.Service.WEB_CLOTHES: 60,
        mt.Service.WEB_ADULT: 800,
        mt.Service.DAILYMOTION: 800,
        mt.Service.INSTAGRAM: 200,
        mt.Service.CLASH_OF_CLANS: 100,
        mt.Service.POKEMON_GO: 60,
        mt.Service.WEB_FINANCE: 50,
        mt.Service.FACEBOOK_LIVE: 800,
        mt.Service.EA_GAMES: 100,
        mt.Service.APPLE_VIDEO: 900,
        mt.Service.LINKEDIN: 100,
        mt.Service.SNAPCHAT: 200,
        mt.Service.DEEZER: 150,
        mt.Service.NETFLIX: 900,
        mt.Service.FACEBOOK: 200,
        mt.Service.MOLOTOV: 900,
        mt.Service.WEB_E_COMMERCE: 60,
        mt.Service.FORTNITE: 100,
        mt.Service.PERISCOPE: 800,
        mt.Service.SPOTIFY: 150,
        mt.Service.WEB_STREAMING: 800,
        mt.Service.YAHOO: 100
    }

    samplers = {s: get_normal_distribution_sampler(mean=hourly_traffic_mb[s] / 60, std=0.5 * hourly_traffic_mb[s] / 60) for s in hourly_traffic_mb}
    return samplers


def thresholds_and_buffers_amenities():
    thresholds_and_buffers = [
        (0, 500),
        (0, 200),
        (3, 1000),
        (3, 500),
        (3, 200),
        (6, 1000),
        (6, 500),
    ]
    return thresholds_and_buffers


if __name__ == '__main__':
    pass