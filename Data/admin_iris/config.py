from enum import Enum

data_folder_path = '/Users/anmusso/Desktop/PhD/NetMob/NetMobData/data/AdminData'
data_file = 'AdminComplete.csv'
metadata_file = 'AdminCompleteMetaData.csv'


def get_data_file_path() -> str:
    return f'{data_folder_path}/{data_file}'


def get_metadata_file_path() -> str:
    return f'{data_folder_path}/{metadata_file}'


class INSEEDataFileName(Enum):
    OCCUPATION = 'ActiviteResidents.csv'
    COUPLES_FAMILIES_HOUSEHOLDS = 'CouplesFamillesMenages.csv'
    EDUCATION = 'Diplomes.csv'
    EQUIPMENT_COUNTS = 'EquipementsDenombrement.csv'
    HOUSING = 'Logement.csv'
    POPULATION = 'Population.csv'
    INCOME = 'RevenuPauvrete.csv'


def get_insee_data_file_path(file_name: INSEEDataFileName, year: int) -> str:
    return f'{data_folder_path}/{year}/{file_name.value}'


def get_insee_metadata_file_path(file_name: INSEEDataFileName, year: int) -> str:
    return f'{data_folder_path}/{year}/{file_name.value.replace(".csv", "MetaData.csv")}'