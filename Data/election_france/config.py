data_folder = '/Users/anmusso/Desktop/PhD/NetMob/NetMobData/data/ElectionData'
raw_data_file = 'resultats-definitifs-par-bureau-de-vote_europeens_2019.csv'
clean_data_file = 'EuropeanElectionResults2019_PollingStationLevel.csv'


def get_data_file():
    return f'{data_folder}/{clean_data_file}'


def get_raw_data_file():
    return f'{data_folder}/{raw_data_file}'
