from datetime import date
import itertools
from joblib import Parallel, delayed

import pandas as pd
from tqdm import tqdm

from Utils import City, Service, TrafficType, AggregationLevel
from DataIO import DataIO
from IrisAggregation.IrisTileMatching import IrisTileMatchingAPI, IrisTileMatching


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


def rename_iris_to_str():
    from DataIO import DataIO
    from Utils import City, Service, TrafficType, AggregationLevel
    import itertools
    from tqdm import tqdm
    iris_codes = set(IrisTileMatchingAPI.load_matching().data['iris'].unique())

    def iris_int_to_str(iris_int: int) -> str:
        iris_str = str(iris_int)
        if len(iris_str) == 9 and iris_str in iris_codes:
            return iris_str
        elif len(iris_str) == 8:
            iris_str = '0' + iris_str
            if iris_str in iris_codes:
                return iris_str
            else:
                raise ValueError(f'Invalid iris code: {iris_int}')
        else:
            raise ValueError(f'Invalid iris code: {iris_int}')

    specs = list(itertools.product(Service, TrafficType, DataIO.get_days()))
    cities = [city for city in City][8:]
    for city in tqdm(cities):
        location_ids = DataIO.get_location_ids(aggregation_level=AggregationLevel.IRIS, city=city)
        for service, traffic_type, day in specs:
            data = DataIO._load_traffic_data_base(city=city, service=service, traffic_type=traffic_type, day=day, aggregation_level=AggregationLevel.IRIS)
            if len(data) > len(location_ids):
                data_new = data.iloc[:len(location_ids)]
            else:
                data_new = data

            data_new.index = [iris_int_to_str(iris_int=iris_int) for iris_int in data_new.index]
            DataIO.save_iris_aggregated_traffic_data(data=data_new, traffic_type=traffic_type, city=city, service=service, day=day)


if __name__ == '__main__':
    rename_iris_to_str()





