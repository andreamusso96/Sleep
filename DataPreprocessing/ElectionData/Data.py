from typing import List

import numpy as np
import pandas as pd

from config import ELECTION_DATA_PATH


class ElectionDataRaw:
    def __init__(self):
        self.file_path = f'{ELECTION_DATA_PATH}/resultats-definitifs-par-bureau-de-vote_europeens_2019.csv'
        self.data = self.load()

    def load(self):
        data = pd.read_csv(self.file_path, low_memory=False, index_col=0)
        data = self._reformat(data)
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
    def _reformat(data):
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


class ElectionData:
    def __init__(self):
        self.file_path = f'{ELECTION_DATA_PATH}/european_elections_2019_results_by_polling_station.csv'
        self.data = self.load()

    def load(self):
        data = pd.read_csv(self.file_path, low_memory=False)
        data['municipality_code'] = data['department_code'].astype(str).str.zfill(2) + data['municipality_code'].astype(str).str.zfill(3)
        data['polling_station'] = data['municipality_code'] + data['polling_station_code']
        data.drop(columns=['department_code', 'department_name'], inplace=True)
        return data


if __name__ == '__main__':
    e = ElectionData()