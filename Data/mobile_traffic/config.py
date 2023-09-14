from .enums import City, Service, TrafficType, GeoDataType
from datetime import date

data_folder = '/Users/anmusso/Desktop/PhD/NetMob/NetMobData/data/TrafficData'
matching_iris_tile_file = 'MatchingIrisPollingStation.csv'


def get_matching_iris_tile_file_path():
    return f'{data_folder}/{matching_iris_tile_file}'


def get_mobile_traffic_data_file_path(traffic_type: TrafficType, geo_data_type: GeoDataType, city: City, service: Service, day: date):
    day_str = day.strftime('%Y%m%d')
    path = f'{data_folder}/{geo_data_type.value}/{city.value}/{service.value}/{day_str}/'
    file_name = f'{city.value}_{service.value}_{day_str}_{traffic_type.value}.txt'
    file_path = path + file_name
    return file_path