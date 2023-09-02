data_folder = '/Users/anmusso/Desktop/PhD/NetMob/NetMobData/data/ElectionData'
raw_data_file = 'resultats-definitifs-par-bureau-de-vote_europeens_2019.csv'
clean_data_file = 'EuropeanElectionResults2019_PollingStationLevel.csv'
matching_iris_polling_station_file = 'MatchingIrisPollingStation.csv'


def get_data_file_path():
    return f'{data_folder}/{clean_data_file}'


def get_raw_data_file_path():
    return f'{data_folder}/{raw_data_file}'


def get_matching_iris_polling_station_file_path():
    return f'{data_folder}/{matching_iris_polling_station_file}'
