from typing import List

import pandas as pd

from DataPreprocessing.AdminData.Data import Activite, CouplesFamilleMenage, Diplomes, Equipements, Logement, Population, Revenus, AdminData
from DataPreprocessing.GeoData.GeoDataType import GeoDataType


class Merger:
    def __init__(self):
        self.activite = Activite()
        self.couples_famille_menage = CouplesFamilleMenage()
        self.diplomes = Diplomes()
        self.equipements = Equipements()
        self.logement = Logement()
        self.population = Population()
        self.revenus = Revenus()
        self.datasets: List[AdminData] = [self.activite, self.couples_famille_menage, self.diplomes, self.equipements,
                                          self.logement, self.population, self.revenus]

    def load_datasets(self):
        for dataset in self.datasets:
            dataset.load()

    def load_meta_data(self):
        for dataset in self.datasets:
            if not isinstance(dataset, Equipements):
                dataset.load_metadata()

    def merge(self) -> pd.DataFrame:
        merged_dataset = self._merge(dataset1=self.activite.data, dataset2=self.couples_famille_menage.data)
        merged_dataset = self._merge(dataset1=merged_dataset, dataset2=self.diplomes.data)
        equipements_iris_by_type = self.equipements.get_equipment_data_iris_by_type()
        merged_dataset = self._merge(dataset1=merged_dataset, dataset2=equipements_iris_by_type)
        merged_dataset = self._merge(dataset1=merged_dataset, dataset2=self.logement.data)
        merged_dataset = self._merge(dataset1=merged_dataset, dataset2=self.population.data)
        merged_dataset = self._merge(dataset1=merged_dataset, dataset2=self.revenus.data)
        return merged_dataset

    def merge_metadata(self) -> pd.DataFrame:
        merged_metadata = pd.concat([dataset.metadata[['COD_VAR', 'LIB_VAR_LONG']] for dataset in self.datasets if
                                     not isinstance(dataset, Equipements)])
        rows_to_drop = merged_metadata['COD_VAR'].isin(['IRIS', 'COM', 'MODIF_IRIS', 'LAB_IRIS'])
        merged_metadata = merged_metadata[~rows_to_drop]
        merged_metadata = merged_metadata[~merged_metadata['COD_VAR'].duplicated()]
        merged_metadata = merged_metadata.sort_values(by='COD_VAR').reset_index(drop=True)
        return merged_metadata

    @staticmethod
    def _merge(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> pd.DataFrame:
        merged_datasets = dataset1.merge(dataset2, on=GeoDataType.IRIS.value, how='outer', suffixes=('', f'_duplicate'))
        merged_datasets.drop(columns=[c for c in merged_datasets.columns if c.endswith('_duplicate')], inplace=True)
        return merged_datasets


if __name__ == '__main__':
    pass