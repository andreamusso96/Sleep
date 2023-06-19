from enum import Enum

import pandas as pd

from config import ADMIN_DATA_PATH


class AdminDataFileName(Enum):
    ALL = 'AdminComplete.csv'
    ACTIVITE = 'ActiviteResidents.csv'
    COUPLES_FAMILLE_MENAGE = 'CouplesFamillesMenages.csv'
    DIPLOMES = 'Diplomes.csv'
    EQUIPEMENTS = 'EquipementsDenombrement.csv'
    LOGEMENT = 'Logement.csv'
    POPULATION = 'Population.csv'
    REVENUS = 'RevenuPauvrete.csv'


class AdminData:
    def __init__(self, file_name: AdminDataFileName, year: int):
        self.file_name = file_name
        self.year = year
        self.file_path = f'{ADMIN_DATA_PATH}/{self.year}/{self.file_name.value}'
        metadata_file_name = self.file_name.value.replace('.csv', 'MetaData.csv')
        self.metadata_file_path = f'{ADMIN_DATA_PATH}/{self.year}/{metadata_file_name}'
        self.data = None
        self.metadata = None

    def load(self):
        raise NotImplementedError

    def load_metadata(self):
        raise NotImplementedError


class AdminDataBI(AdminData):
    def __init__(self, file_name: AdminDataFileName, year: int):
        super().__init__(file_name=file_name, year=year)
        self.dtype = {'IRIS': str, 'COM': str, 'LAB_IRIS': str}
        self.cols_to_drop = ['COM', 'LAB_IRIS', 'MODIF_IRIS']
        self.low_memory = False

    def load(self):
        data = pd.read_csv(self.file_path, sep=';', dtype=self.dtype, low_memory=self.low_memory)
        data.rename(columns={'IRIS': 'iris'}, inplace=True)
        data.drop(columns=self.cols_to_drop, inplace=True)
        data.sort_values(by='iris', inplace=True)
        self.data = data
        return data

    def load_metadata(self):
        metadata = pd.read_csv(self.metadata_file_path, sep=';')
        self.metadata = metadata
        return metadata


class Activite(AdminDataBI):
    def __init__(self, year: int = 2019):
        super().__init__(file_name=AdminDataFileName.ACTIVITE, year=year)


class CouplesFamilleMenage(AdminDataBI):
    def __init__(self, year: int = 2019):
        super().__init__(AdminDataFileName.COUPLES_FAMILLE_MENAGE, year=year)


class Diplomes(AdminDataBI):
    def __init__(self, year: int = 2019):
        super().__init__(AdminDataFileName.DIPLOMES, year=year)


class Equipements(AdminData):
    def __init__(self, year: int = 2021):
        super().__init__(AdminDataFileName.EQUIPEMENTS, year=year)
        self.dtype = {'DCIRIS': str}
        self.cols_to_use = ['DCIRIS', 'DOM', 'SDOM', 'TYPEQU', 'NB_EQUIP']

    def load(self):
        data = pd.read_csv(self.file_path, sep=';', usecols=self.cols_to_use, dtype=self.dtype)
        data = data[self.cols_to_use]
        data.rename(columns={'DCIRIS': 'iris'}, inplace=True)
        data = data.groupby(by=['iris', 'TYPEQU']).sum().reset_index()
        data.sort_values(by='iris', inplace=True)
        self.data = data
        return data

    def get_equipment_data_iris_by_type(self, prefix: str = 'EQUIP_'):
        assert self.data is not None, 'Data not loaded'
        equipements_iris_by_type = self.data.pivot(index='iris', columns='TYPEQU', values='NB_EQUIP').fillna(0).reset_index()
        equipements_iris_by_type.rename(columns={col: f'{prefix}{col}' for col in equipements_iris_by_type.columns if col != 'iris'}, inplace=True)
        return equipements_iris_by_type


class Logement(AdminDataBI):
    def __init__(self, year: int = 2019):
        super().__init__(AdminDataFileName.LOGEMENT, year=year)


class Population(AdminDataBI):
    def __init__(self, year: int = 2019):
        super().__init__(AdminDataFileName.POPULATION, year=year)


class Revenus(AdminData):
    def __init__(self, year: int = 2019):
        super().__init__(AdminDataFileName.REVENUS, year=year)
        self.dtype = {'IRIS': str}

    def load(self):
        data = pd.read_csv(self.file_path, sep=',', dtype=self.dtype)
        data.rename(columns={'IRIS': 'iris'}, inplace=True)
        data['iris'] = data['iris'].apply(self._reformat_iris_string)
        data.sort_values(by='iris', inplace=True)
        self.data = data
        return data

    def load_metadata(self):
        metadata = pd.read_csv(self.metadata_file_path, sep=';')
        self.metadata = metadata
        return metadata

    @staticmethod
    def _reformat_iris_string(iris) -> str:
        if len(iris) == 9:
            return iris
        elif len(iris) == 8:
            return f'0{iris}'
        else:
            raise ValueError(f'Invalid iris: {iris}')


class SelectedPopulationVariables:
    def __init__(self):
        self.file_path = f'{ADMIN_DATA_PATH}/selected_population_variables.csv'
        self.data = pd.read_excel(f'{ADMIN_DATA_PATH}/selected_population_variables.xlsx', header=0)

    def get_selected_population_variables(self):
        return list(self.data['COD_VAR'])


class AdminDataComplete:
    def __init__(self, year: int = 2019):
        self.year = year
        self.file_path = f'{ADMIN_DATA_PATH}/{self.year}/{AdminDataFileName.ALL.value}'
        self.pop_metadata_file_path = f'{ADMIN_DATA_PATH}/{self.year}/{AdminDataFileName.ALL.value.replace(".csv", "MetaData.csv")}'
        self.equip_metadata_file_path = f'{ADMIN_DATA_PATH}/{2021}/{AdminDataFileName.EQUIPEMENTS.value.replace("Denombrement.csv", "Classification.csv")}'
        self.selected_pop_vars_file_path = f'{ADMIN_DATA_PATH}/selected_population_variables.csv'
        self.data = self.load()
        self.pop_metadata, self.equip_metadata, self.selected_pop_vars = self.load_metadata()

    def load(self):
        data = pd.read_csv(self.file_path, dtype={'IRIS': str}, low_memory=False)
        data.sort_values(by='iris', inplace=True)
        data.set_index('iris', inplace=True)
        return data

    def load_metadata(self):
        pop_metadata = pd.read_csv(self.pop_metadata_file_path)
        equip_metadata = pd.read_csv(self.equip_metadata_file_path)
        selected_pop_vars = pd.read_excel(f'{ADMIN_DATA_PATH}/selected_population_variables.xlsx', header=0)
        return pop_metadata, equip_metadata, selected_pop_vars