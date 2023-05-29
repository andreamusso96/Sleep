import pandas as pd

from config import ADMIN_DATA_PATH

class PopulationData:
    def __init__(self):
        self.data_file_path = f'{ADMIN_DATA_PATH}/PopulationData2019.csv'
        self.meta_data_file_path = f'{ADMIN_DATA_PATH}/PopulationMetaData.csv'
        self.data, self.meta_data = self.load()

    def load(self):
        dtypes = {'IRIS': str, 'COM': str, 'LAB_IRIS': str}
        data = pd.read_csv(self.data_file_path, sep=';', dtype=dtypes)
        data.rename(columns={'IRIS': 'iris'}, inplace=True)
        meta_data = pd.read_csv(self.meta_data_file_path, sep=';', dtype={'IRIS': str})
        return data, meta_data


class BusinessClassificationData:
    def __init__(self):
        self.file_path = f'{ADMIN_DATA_PATH}/BusinessClassification.csv'
        self.data = self.load()

    def load(self):
        return pd.read_csv(self.file_path)


class BusinessCountsData:
    def __init__(self):
        self.file_path = f'{ADMIN_DATA_PATH}/BusinessCounts2021.csv'
        self.data = self.load()

    def load(self):
        cols_to_use = ['DCIRIS', 'DOM', 'SDOM', 'TYPEQU', 'NB_EQUIP']
        dtypes = {'DCIRIS': str}
        data = pd.read_csv(self.file_path, sep=';', usecols=cols_to_use, dtype=dtypes)
        data = data[cols_to_use]
        data.rename(columns={'DCIRIS': 'iris'}, inplace=True)
        return data


class BusinessDensityIndex:
    def __init__(self, business_counts: BusinessCountsData, population: PopulationData):
        self.business_counts = business_counts
        self.population = population

    def compute_index(self):
        number_of_businesses_by_iris = self.business_counts.data.groupby('iris')['NB_EQUIP'].sum()
        population_in_iris = self.population.data.set_index('iris')['P19_POP'].astype(int)
        number_of_businesses_and_population = pd.merge(number_of_businesses_by_iris, population_in_iris, left_index=True, right_index=True)
        business_density_index = (number_of_businesses_and_population['NB_EQUIP'] / number_of_businesses_and_population['P19_POP']).to_frame().clip(lower=0, upper=1).rename(columns={0: 'business_density_index'})
        return business_density_index


if __name__ == '__main__':
    business_density_index = BusinessDensityIndex(BusinessCountsData(), PopulationData())
    business_density_index.compute_index()

