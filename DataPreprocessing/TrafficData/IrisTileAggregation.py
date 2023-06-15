from datetime import date
import itertools
from joblib import Parallel, delayed

import pandas as pd

from Utils import City, Service, TrafficType
from DataIO import DataIO
from DataPreprocessing.GeoData.GeoMatching import GeoMatchingAPI, GeoMatching
from DataPreprocessing.GeoData.GeoDataType import GeoDataType


class Aggregator:
    def __init__(self, geo_matching: GeoMatching):
        self.geo_matching = geo_matching

    def aggregate_traffic_data(self):
        city_service_day_combinations = list(itertools.product(City, Service, DataIO.get_days()))
        Parallel(n_jobs=-1, verbose=1)(delayed(self._aggregate_and_save_traffic_city_service_day)(city=city, service=service, day=day) for city, service, day in city_service_day_combinations)

    def _aggregate_and_save_traffic_city_service_day(self, city: City, service: Service, day: date):
        aggregated_ul_data = self._aggregate_traffic_data_file(city=city, service=service, traffic_type=TrafficType.UL, day=day)
        aggregated_dl_data = self._aggregate_traffic_data_file(city=city, service=service, traffic_type=TrafficType.DL, day=day)
        DataIO.save_iris_aggregated_traffic_data(data=aggregated_ul_data, traffic_type=TrafficType.UL, city=city,
                                                 service=service, day=day)
        DataIO.save_iris_aggregated_traffic_data(data=aggregated_dl_data, traffic_type=TrafficType.DL, city=city,
                                                 service=service, day=day)

    def _aggregate_traffic_data_file(self, traffic_type: TrafficType, city: City, service: Service, day: date) -> pd.DataFrame:
        data = DataIO.load_traffic_data(traffic_type=traffic_type, geo_data_type=GeoDataType.TILE, city=city,
                                        service=service, day=day).to_pandas()
        data[GeoDataType.IRIS.value] = data.apply(lambda row: self.geo_matching.get_iris(city=city, tile=row.name), axis=1)
        data = data.groupby(by=GeoDataType.TILE.value).sum()
        return data


class IrisTileAggregationAPI:
    @staticmethod
    def aggregate_traffic_data():
        geo_matching = GeoMatchingAPI.load_matching()
        aggregator = Aggregator(geo_matching=geo_matching)
        aggregator.aggregate_traffic_data()





