data_folder = '/Users/anmusso/Desktop/PhD/NetMob/NetMobData/data/GeoData/PollingStationGeo/'
raw_data_file = 'bureaux-vote-france-2017.geojson'
clean_data_file = 'PollingStationGeo.geojson'
crosswalk_parent_municipality_code_to_municipality_code = 'commune_codes-01012019.csv'


def get_data_file_path():
    return f'{data_folder}/{clean_data_file}'


def get_raw_data_file_path():
    return f'{data_folder}/{raw_data_file}'


def get_crosswalk_parent_municipality_code_to_municipality_code_file_path():
    return f'{data_folder}/{crosswalk_parent_municipality_code_to_municipality_code}'