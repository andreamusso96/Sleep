from datetime import date
from typing import Tuple
import itertools

import pandas as pd
from tqdm import tqdm

from Utils import City, Service, TrafficType, AggregationLevel
from DataIO import DataIO
from IrisAggregation.IrisTileMatching import IrisTileMatchingAPI, IrisTileMatching


class Aggregator:
    def __init__(self, iris_tile_matching: IrisTileMatching):
        self.iris_tile_matching = iris_tile_matching

    def aggregate_traffic_data(self):
        city_service_day_combinations = itertools.product(City, Service, DataIO.get_days())
        for city, service, day in tqdm(city_service_day_combinations):
            aggregated_ul_data, aggregated_dl_data = self._aggregate_traffic_city_service_day(city=city, service=service, day=day)
            DataIO.save_iris_aggregated_traffic_data(data=aggregated_ul_data, traffic_type=TrafficType.UL, city=city, service=service, day=day)
            DataIO.save_iris_aggregated_traffic_data(data=aggregated_dl_data, traffic_type=TrafficType.DL, city=city, service=service, day=day)

    def _aggregate_traffic_city_service_day(self, city: City, service: Service, day: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        aggregated_ul_data = self._aggregate_traffic_data_file(city=city, service=service, traffic_type=TrafficType.UL, day=day)
        aggregated_dl_data = self._aggregate_traffic_data_file(city=city, service=service, traffic_type=TrafficType.DL, day=day)
        return aggregated_ul_data, aggregated_dl_data

    def _aggregate_traffic_data_file(self, traffic_type: TrafficType, city: City, service: Service, day: date) -> pd.DataFrame:
        data = DataIO.load_traffic_data(city=city, service=service, traffic_type=traffic_type, day=day, aggregation_level=AggregationLevel.TILE).to_pandas()
        data['iris_code'] = data.apply(lambda row: self.iris_tile_matching.get_iris_code(city=city, tile_id=row.name), axis=1)
        data = data.groupby(by='iris_code').sum()
        return data


class IrisTileAggregationAPI:
    @staticmethod
    def aggregate_traffic_data():
        iris_tile_matching = IrisTileMatchingAPI.load_matching()
        aggregator = Aggregator(iris_tile_matching=iris_tile_matching)
        aggregator.aggregate_traffic_data()


def test():
    c = City.LYON
    s = Service.YOUTUBE
    d = date(2019, 3, 16)
    tt = TrafficType.DL
    agg = Aggregator(iris_tile_matching=IrisTileMatchingAPI.load_matching())
    data = agg._aggregate_traffic_data_file(city=c, service=s, traffic_type=tt, day=d)
    DataIO.save_iris_aggregated_traffic_data(data=data, traffic_type=tt, city=c, service=s, day=d)




