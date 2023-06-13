import os.path
from datetime import date
from joblib import Parallel, delayed

import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd

from config import TRAFFIC_DATA_PATH, GEO_DATA_PATH
from Utils import TrafficType, City, Service, AggregationLevel, TrafficDataDimensions


class DataIO:
    @staticmethod
    def load_traffic_data(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City = None, service: Service=None, day: date=None) -> xr.DataArray:
        if service is None and day is None and city is not None:
            return DataIO._city_traffic_data(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city)
        elif service is None and day is not None and city is not None:
            return DataIO._city_day_traffic_data(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, day=day)
        elif service is not None and day is None and city is not None:
            return DataIO._city_service_traffic_data(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service)
        elif service is not None and day is not None and city is not None:
            return DataIO._city_service_day_traffic_data(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day)
        elif service is not None and day is None and city is None:
            return DataIO._service_traffic_data(traffic_type=traffic_type, aggregation_level=aggregation_level, service=service)
        else:
            raise ValueError(f'Invalid parameters for DataIO.load_traffic_data: traffic_type={traffic_type}, aggregation_level={aggregation_level}, city={city}, service={service}, day={day}')

    @staticmethod
    def save_iris_aggregated_traffic_data(data: pd.DataFrame, traffic_type: TrafficType, city: City, service: Service, day: date):
        file_name = DataIO._get_traffic_data_file_path(traffic_type=traffic_type, city=city, service=service, day=day, aggregation_level=AggregationLevel.IRIS)
        file_directory = os.path.dirname(file_name)
        if not os.path.exists(file_directory):
            os.makedirs(file_directory)
        data.sort_index(inplace=True)
        data.to_csv(path_or_buf=file_name, sep=' ', index=True, header=False)

    @staticmethod
    def _city_traffic_data(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City) -> xr.DataArray:
        data_vals = []
        days = DataIO.get_days()
        for day in tqdm(days):
            data_vals_day = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day) for service in Service.get_services(traffic_type=traffic_type))
            data_vals.append(np.stack(data_vals_day, axis=-1))

        data = np.stack(data_vals, axis=-1)
        coords = {aggregation_level.value: DataIO.get_location_ids(aggregation_level=aggregation_level, city=city),
                  TrafficDataDimensions.TIME.value: DataIO.get_times(),
                  TrafficDataDimensions.SERVICE.value: Service.get_services(traffic_type=traffic_type, return_values=True),
                  TrafficDataDimensions.DAY.value: DataIO.get_days()}
        dims = [aggregation_level.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.SERVICE.value, TrafficDataDimensions.DAY.value]
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_service_traffic_data(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City, service: Service) -> xr.DataArray:
        days = DataIO.get_days()
        data_vals = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day) for day in days)
        data = np.stack(data_vals, axis=-1)
        coords = {aggregation_level.value: DataIO.get_location_ids(aggregation_level=aggregation_level, city=city),
                  TrafficDataDimensions.TIME.value: DataIO.get_times(),
                  TrafficDataDimensions.DAY.value: DataIO.get_days()}
        dims = [aggregation_level.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.DAY.value]
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_day_traffic_data(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City, day: date) -> xr.DataArray:
        data_vals = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day) for service in Service.get_services(traffic_type=traffic_type))
        data = np.stack(data_vals, axis=-1)
        coords = {aggregation_level.value: DataIO.get_location_ids(aggregation_level=aggregation_level, city=city),
                  TrafficDataDimensions.TIME.value: DataIO.get_times(),
                  TrafficDataDimensions.SERVICE.value: Service.get_services(traffic_type=traffic_type)}
        dims = [aggregation_level.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.SERVICE.value]
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_service_day_traffic_data(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City, service: Service, day: date) -> xr.DataArray:
        data = DataIO._load_traffic_data_base(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day)
        coords = {aggregation_level.value: DataIO.get_location_ids(aggregation_level=aggregation_level, city=city),
                  TrafficDataDimensions.TIME.value: DataIO.get_times()}
        dims = [aggregation_level.value, TrafficDataDimensions.TIME.value]
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _service_traffic_data(traffic_type: TrafficType, aggregation_level: AggregationLevel, service: Service) -> xr.DataArray:
        data_vals = []
        location_ids = []
        for city in tqdm(City):
            data_vals_day = Parallel(n_jobs=-1)(delayed(DataIO._load_traffic_data_base)(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day) for day in DataIO.get_days())
            data_vals.append(np.stack(data_vals_day, axis=-1))
            location_ids += DataIO.get_location_ids(aggregation_level=aggregation_level, city=city)

        data = np.concatenate(data_vals, axis=0)
        coords = {aggregation_level.value: location_ids,
                  TrafficDataDimensions.TIME.value: DataIO.get_times(),
                  TrafficDataDimensions.DAY.value: DataIO.get_days()}
        dims = [aggregation_level.value, TrafficDataDimensions.TIME.value, TrafficDataDimensions.DAY.value]
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _load_traffic_data_base(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City, service: Service, day: date) -> pd.DataFrame:
        if traffic_type == TrafficType.UL_AND_DL or traffic_type == TrafficType.USERS:
            ul_data = DataIO._load_traffic_data_file(traffic_type=TrafficType.UL, aggregation_level=aggregation_level, city=city, service=service, day=day)
            dl_data = DataIO._load_traffic_data_file(traffic_type=TrafficType.DL, aggregation_level=aggregation_level, city=city, service=service, day=day)
            traffic = ul_data + dl_data
            if traffic_type == TrafficType.USERS:
                service_data_consumption = Service.get_service_data_consumption(service=service)
                traffic = traffic / service_data_consumption
                return traffic
            else:
                return traffic
        else:
            return DataIO._load_traffic_data_file(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day)

    @staticmethod
    def _load_traffic_data_file(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City, service: Service, day: date) -> pd.DataFrame:
        file_path = DataIO._get_traffic_data_file_path(traffic_type=traffic_type, aggregation_level=aggregation_level, city=city, service=service, day=day)
        cols = [aggregation_level.value] + list(DataIO.get_times())
        traffic_data = pd.read_csv(file_path, sep=' ', names=cols)
        traffic_data.set_index(aggregation_level.value, inplace=True)
        return traffic_data

    @staticmethod
    def _get_traffic_data_file_path(traffic_type: TrafficType, aggregation_level: AggregationLevel, city: City, service: Service, day: date) -> str:
        day_str = day.strftime('%Y%m%d')
        path = f'{TRAFFIC_DATA_PATH}/{aggregation_level.value}/{city.value}/{service.value}/{day_str}/'
        file_name = f'{city.value}_{service.value}_{day_str}_{traffic_type.value}.txt'
        file_path = path + file_name
        return file_path

    @staticmethod
    def get_times():
        return pd.timedelta_range(start='00:00:00', end='23:59:00', freq='15min')

    @staticmethod
    def get_days():
        return pd.date_range(start='2019-03-16', end='2019-05-31', freq='D')

    @staticmethod
    def get_location_ids(aggregation_level: AggregationLevel, city: City):
        location_code_file = f'{GEO_DATA_PATH}/IrisTileMatching.csv'
        location_codes = pd.read_csv(location_code_file, sep=',', dtype={AggregationLevel.IRIS.value: str, AggregationLevel.TILE.value: str})
        location_codes = location_codes[location_codes['city'] == city.value]
        location_codes = list(np.sort(location_codes[aggregation_level.value].unique()))
        return location_codes


if __name__ == '__main__':
    d = DataIO.load_traffic_data(traffic_type=TrafficType.USERS, aggregation_level=AggregationLevel.IRIS, city=City.LYON)
