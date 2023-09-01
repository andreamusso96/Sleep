import pandas as pd

from . import config


def save_preprocessed_election_data():
    election_data = preprocess_election_data()
    election_data.to_csv(config.get_data_file_path(), index=False)


def preprocess_election_data():
    data = read_election_data()
    data = reformat_data_to_long_format(data=data)
    data = rename_columns(data=data)
    data = set_parent_municipality_code(data=data)
    data = set_polling_station_code(data=data)
    data = drop_redundant_geo_information(data=data)
    data.sort_values(by='polling_station', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def read_election_data() -> pd.DataFrame:
    file_path = config.get_raw_data_file_path()
    data = pd.read_csv(file_path, low_memory=False, index_col=0)
    return data


def reformat_data_to_long_format(data: pd.DataFrame) -> pd.DataFrame:
    original_column_names = list(data.columns)
    index_start_repeated_columns = original_column_names.index('N°Liste')
    index_end_titles_of_repeated_columns = original_column_names.index('Unnamed: 26')
    repeated_column_names = original_column_names[index_start_repeated_columns:index_end_titles_of_repeated_columns]
    n_repeated_columns = len(repeated_column_names)
    n_repetitions = int(len(original_column_names[index_start_repeated_columns:]) / n_repeated_columns)
    chunks = [data[original_column_names[index_start_repeated_columns + i * n_repeated_columns:index_start_repeated_columns + (i + 1) * n_repeated_columns]] for i in range(n_repetitions - 1)]
    chunks = [chunk.rename(columns={c: repeated_column_names[i] for i, c in enumerate(chunk.columns)}) for chunk in chunks]
    concatenated_chunks = pd.concat(chunks, axis=0, ignore_index=False)
    data = concatenated_chunks.merge(data[original_column_names[:index_start_repeated_columns]], left_index=True, right_index=True, how='left')
    data = data[original_column_names[:index_start_repeated_columns] + repeated_column_names]
    data = data.sort_values(by=['Code du département', 'Code de la commune', 'Code du b.vote', 'N°Liste'])
    return data


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
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


def set_parent_municipality_code(data: pd.DataFrame) -> pd.DataFrame:
    data['parent_municipality_code'] = data['department_code'].astype(str).str.zfill(2) + data['municipality_code'].astype(str).str.zfill(3)
    return data


def set_polling_station_code(data: pd.DataFrame) -> pd.DataFrame:
    data['polling_station'] = data['parent_municipality_code'] + data['polling_station_code']
    return data


def drop_redundant_geo_information(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns=['department_code', 'department_name', 'municipality_code', 'polling_station_code'], inplace=True)
    return data


