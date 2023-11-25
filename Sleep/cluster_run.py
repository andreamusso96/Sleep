import sys

import mobile_traffic as mt
import pandas as pd
import xarray as xr
from datetime import datetime

from aggregate_geo import aggregate_netmob_tile_level_traffic_data_city_to_insee_tile_level
import transform
from mobile_data import MobileData, TrafficData, ScreenTimeData
import robustness


def aggregate_and_save_netmob_tile_to_insee_tile(folder_path: str, city: mt.City):
    file_path = f'{folder_path}/mobile_traffic_{city.value.lower()}_by_netmob_tile_service_and_time.nc'
    data_city = load_netmob_tile_dataset_city(file_path=file_path)
    agg = aggregate_netmob_tile_level_traffic_data_city_to_insee_tile_level(traffic_data_netmob_tile=data_city, city=city)
    time_as_str = [str(t) for t in agg.time.values]
    agg = agg.assign_coords(time=time_as_str)
    agg.to_netcdf(f'{folder_path}/mobile_traffic_{city.value.lower()}_by_insee_tile_service_and_time.nc')


def load_netmob_tile_dataset_city(file_path: str):
    data_city = xr.open_dataset(file_path).to_array().squeeze()
    data_city = data_city.assign_coords(time=[datetime.strptime(t, '%H:%M:%S').time() for t in data_city.time.values])
    return data_city


def get_base_data(folder_data: str):
    income_quantiles = [0.3, 0.7]
    n_sample_screen_time_robustness = 15
    traffic_data = TrafficData.load_dataset(synthetic=False, insee_tiles=True, folder_path=folder_data)
    screen_time_data = robustness.screen_time_data_sample(traffic_data=traffic_data, traffic_per_minute_sampler=robustness.service_traffic_per_minute_sampler())
    return traffic_data, screen_time_data, income_quantiles, n_sample_screen_time_robustness


def generate_and_save_other_figure_data(folder_save: str, folder_data: str):
    traffic_data, screen_time_data, income_quantiles, n_sample_screen_time_robustness = get_base_data(folder_data=folder_data)

    nsi_income_tile_geo = transform.night_screen_index_income_category_and_tile_geo(screen_time_data=screen_time_data, income_quantiles=income_quantiles)
    nsi_for_services = transform.night_screen_index_for_services(screen_time_data=screen_time_data)
    rca_income_time = transform.rca_income_and_time(screen_time_data=screen_time_data, income_quantiles=income_quantiles)

    nsi_income_tile_geo.to_file(f'{folder_save}/nsi_income_tile_geo.geojson', driver='GeoJSON')
    nsi_for_services.to_csv(f'{folder_save}/nsi_for_services.csv', index=False)
    rca_income_time.to_csv(f'{folder_save}/rca_income_time.csv')


def generate_and_save_nsi_robustness_checks_data(folder_save: str, folder_data: str):
    traffic_data, screen_time_data, income_quantiles, n_sample_screen_time_robustness = get_base_data(folder_data=folder_data)

    nsi_income = transform.night_screen_index_and_log2_income(screen_time_data=screen_time_data)
    nsi_income_robustness__screen_time_data = transform.night_screen_index_and_log2_income_robustness__screen_time(traffic_data=traffic_data, income_quantiles=income_quantiles, n_samples=n_sample_screen_time_robustness)
    nsi_income_robustness__amenity_data = transform.night_screen_index_and_log2_income_robustness__amenity(screen_time_data=screen_time_data, income_quantiles=income_quantiles)

    nsi_income.to_csv(f'{folder_save}/nsi_income.csv')
    nsi_income_robustness__screen_time_data.to_csv(f'{folder_save}/nsi_income_robustness__screen_time.csv')
    nsi_income_robustness__amenity_data.to_csv(f'{folder_save}/nsi_income_robustness__amenity.csv')


def generate_and_save_rca_robustness_checks_data(folder_save: str, folder_data: str):
    traffic_data, screen_time_data, income_quantiles, n_sample_screen_time_robustness = get_base_data(folder_data=folder_data)

    rca_income_service_data = transform.rca_income_and_services(mobile_data=screen_time_data, income_quantiles=income_quantiles)
    rca_income_service_robustness__screen_time_data = transform.rca_income_and_services_robustness__screen_time(traffic_data=traffic_data, income_quantiles=income_quantiles, n_samples=n_sample_screen_time_robustness)
    rca_income_service_robustness__amenity_data = transform.rca_income_and_services_robustness__amenity(screen_time_data=screen_time_data, income_quantiles=income_quantiles)

    rca_income_service_data.to_csv(f'{folder_save}/rca_income_service.csv')
    rca_income_service_robustness__screen_time_data.to_netcdf(f'{folder_save}/rca_income_service_robustness__screen_time.csv')
    rca_income_service_robustness__amenity_data.to_netcdf(f'{folder_save}/rca_income_service_robustness__amenity.csv')


if __name__ == '__main__':
    arg = sys.argv[1]
    folder = '/cluster/work/gess/coss/users/anmusso/Temp'
    if arg == 'rca':
        generate_and_save_rca_robustness_checks_data(folder_save=folder, folder_data=folder)
    elif arg == 'nsi':
        generate_and_save_nsi_robustness_checks_data(folder_save=folder, folder_data=folder)
    elif arg == 'other':
        generate_and_save_other_figure_data(folder_save=folder, folder_data=folder)