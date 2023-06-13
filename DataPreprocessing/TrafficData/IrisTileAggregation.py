from datetime import date
import itertools
from joblib import Parallel, delayed

import pandas as pd

from Utils import City, Service, TrafficType, AggregationLevel
from DataIO import DataIO
from DataPreprocessing.GeoData.IrisTileMatch.IrisTileMatching import IrisTileMatchingAPI, IrisTileMatching


class Aggregator:
    def __init__(self, iris_tile_matching: IrisTileMatching):
        self.iris_tile_matching = iris_tile_matching

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
        data = DataIO.load_traffic_data(city=city, service=service, traffic_type=traffic_type, day=day, aggregation_level=AggregationLevel.TILE).to_pandas()
        data[AggregationLevel.IRIS.value] = data.apply(lambda row: self.iris_tile_matching.get_iris_code(city=city, tile_id=row.name), axis=1)
        data = data.groupby(by=AggregationLevel.TILE.value).sum()
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
    # DataIO.save_iris_aggregated_traffic_data(data=data, traffic_type=tt, city=c, service=s, day=d)





