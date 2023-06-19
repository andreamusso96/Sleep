from typing import List

import numpy as np
import pandas as pd

from config import ELECTION_DATA_PATH
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
from DataPreprocessing.GeoData.GeoDataComplete import GeoData


class ElectionDataRaw:
    def __init__(self):
        self.file_path = f'{ELECTION_DATA_PATH}/resultats-definitifs-par-bureau-de-vote_europeens_2019.csv'
        self.data = self.load()

    def load(self):
        data = pd.read_csv(self.file_path, low_memory=False, index_col=0)
        data = self._transform_to_long_format(data=data)
        data = self._rename_columns(data=data)
        data = self._set_parent_municipality_code(data=data)
        return data

    @staticmethod
    def _transform_to_long_format(data):
        original_column_names = list(data.columns)
        index_start_repeated_columns = original_column_names.index('N°Liste')
        index_end_titles_of_repeated_columns = original_column_names.index('Unnamed: 26')
        repeated_column_names = original_column_names[index_start_repeated_columns:index_end_titles_of_repeated_columns]
        n_repeated_columns = len(repeated_column_names)
        n_repetitions = int(len(original_column_names[index_start_repeated_columns:]) / n_repeated_columns)
        chunks = [data[original_column_names[index_start_repeated_columns + i*n_repeated_columns:index_start_repeated_columns + (i+1)*n_repeated_columns]] for i in range(n_repetitions - 1)]
        chunks = [chunk.rename(columns={c: repeated_column_names[i] for i, c in enumerate(chunk.columns)}) for chunk in chunks]
        concatenated_chunks = pd.concat(chunks, axis=0, ignore_index=False)
        data = concatenated_chunks.merge(data[original_column_names[:index_start_repeated_columns]],left_index=True, right_index=True, how='left')
        data = data[original_column_names[:index_start_repeated_columns] + repeated_column_names]
        data = data.sort_values(by=['Code du département', 'Code de la commune', 'Code du b.vote', 'N°Liste'])
        return data

    @staticmethod
    def _rename_columns(data):
        column_names = {
            'Code du département': 'department_code',
            'Libellé du département': 'department_name',
            'Code de la commune': 'municipality_code',
            'Libellé de la commune': 'municipality_name',
            'Code du b.vote': 'polling_station_code',
            'Inscrits': 'registered_voters',
            'Abstentions': 'abstentions',
            '% Abs/Ins': 'pct_abstentions',
            'Votants': 'voters',
            '% Vot/Ins': 'pct_voters',
            'Blancs': 'blank_votes',
            '% Blancs/Ins': 'pct_blank_votes',
            '% Blancs/Vot': 'pct_blank_among_votes',
            'Nuls': 'null_votes',
            '% Nuls/Ins': 'pct_null_votes',
            '% Nuls/Vot': 'pct_null_among_votes',
            'Exprimés': 'expressed_votes',
            '% Exp/Ins': 'pct_expressed_votes',
            '% Exp/Vot': 'pct_expressed_among_votes',
            'N°Liste': 'list_number',
            'Libellé Abrégé Liste': 'short_list_label',
            'Libellé Etendu Liste': 'extended_list_label',
            'Nom Tête de Liste': 'list_head_name',
            'Voix': 'votes_to_list',
            '% Voix/Ins': 'pct_votes_to_list',
            '% Voix/Exp': 'pct_votes_to_list_among_votes',
        }
        data.rename(columns=column_names, inplace=True)
        return data

    @staticmethod
    def _set_parent_municipality_code(data):
        data['parent_municipality_code'] = data['department_code'].astype(str).str.zfill(2) + data['municipality_code'].astype(str).str.zfill(3)
        return data


class ElectionData:
    def __init__(self):
        self.file_path = f'{ELECTION_DATA_PATH}/european_elections_2019_results_by_polling_station.csv'
        self.data = self.load()

    def load(self):
        data = pd.read_csv(self.file_path, low_memory=False)
        data[GeoDataType.POLLING_STATION.value] = data['parent_municipality_code'] + data['polling_station_code']
        data.drop(columns=['department_code', 'department_name', 'municipality_code'], inplace=True)
        return data

    def get_election_data_table(self, column: str, value: str):
        table_value_polling_station_by_column = self.data.pivot(index=GeoDataType.POLLING_STATION.value, columns=column, values=value)
        return table_value_polling_station_by_column

    def get_election_data_table_iris_by_column(self, geo_data: GeoData, subset: List[str], column: str, value: str, aggregation_method: str):
        table_value_polling_station_by_column = self.get_election_data_table(column=column, value=value)
        polling_station_iris_map = geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset, other_geo_data_types=GeoDataType.POLLING_STATION).set_index(GeoDataType.POLLING_STATION.value)[GeoDataType.IRIS.value]
        table_value_polling_station_by_column_with_iris_info = table_value_polling_station_by_column.merge(polling_station_iris_map, left_index=True, right_index=True, how='inner')
        table_value_iris_by_column = table_value_polling_station_by_column_with_iris_info.groupby(by=GeoDataType.IRIS.value).agg(aggregation_method)
        return table_value_iris_by_column





if __name__ == '__main__':
    e = ElectionDataRaw()
    e.data.to_csv(f'{ELECTION_DATA_PATH}/european_elections_2019_results_by_polling_station.csv', index=False)