from datetime import date
import itertools
from joblib import Parallel, delayed
import os

import pandas as pd

from . enums import Service, TrafficType, City, TimeOptions, GeoDataType
from . import preprocessing
from . import config
from .match_iris_tile import geo_matching


def aggregate_and_save_traffic_data():
    city_service_day_combinations = list(itertools.product(City, Service, TimeOptions.get_days()))
    Parallel(n_jobs=-1, verbose=1)(delayed(aggregate_and_save_traffic_city_service_day)(city=city, service=service, day=day) for city, service, day in city_service_day_combinations)


def aggregate_and_save_traffic_city_service_day(city: City, service: Service, day: date):
    aggregated_ul_data = aggregate_traffic_data_file(city=city, service=service, traffic_type=TrafficType.UL, day=day)
    aggregated_dl_data = aggregate_traffic_data_file(city=city, service=service, traffic_type=TrafficType.DL, day=day)
    save_iris_aggregated_traffic_data(data=aggregated_ul_data, traffic_type=TrafficType.UL, city=city, service=service, day=day)
    save_iris_aggregated_traffic_data(data=aggregated_dl_data, traffic_type=TrafficType.DL, city=city, service=service, day=day)


def aggregate_traffic_data_file(traffic_type: TrafficType, city: City, service: Service, day: date) -> pd.DataFrame:
    data = preprocessing.load_traffic_data_file(traffic_type=traffic_type, geo_data_type=GeoDataType.TILE, city=city, service=service, day=day)
    data[GeoDataType.IRIS.value] = data.apply(lambda row: geo_matching.get_iris(city=city, tile=row.name), axis=1)
    data = data.groupby(by=GeoDataType.TILE.value).sum()
    return data


def save_iris_aggregated_traffic_data(data: pd.DataFrame, traffic_type: TrafficType, city: City, service: Service, day: date):
    file_name = config.get_mobile_traffic_data_file_path(traffic_type=traffic_type, city=city, service=service, day=day, geo_data_type=GeoDataType.IRIS)
    file_directory = os.path.dirname(file_name)
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    data.sort_index(inplace=True)
    data.to_csv(path_or_buf=file_name, sep=' ', index=True, header=False)
