from typing import Dict

import geopandas as gpd
import pandas as pd

from . import config


def save_preprocessed_polling_station_geo_data():
    data = preprocess_polling_station_geo_data()
    data.to_file(filename=config.get_data_file_path(), driver='GeoJSON')


def preprocess_polling_station_geo_data():
    data = read_polling_station_geo_data_file()
    data = rename_and_select_columns(data=data)
    data = format_polling_station_codes(data=data)
    data = convert_municipality_codes_to_parent_municipality_codes(data=data)
    data = set_polling_station_code(data=data)
    data = drop_redundant_geo_information(data=data)
    data.dropna(subset=['geometry'], inplace=True)
    data.sort_values(by=['polling_station'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def read_polling_station_geo_data_file() -> gpd.GeoDataFrame:
    return gpd.read_file(filename=config.get_raw_data_file_path())


def rename_and_select_columns(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    column_names = {
        'code_insee': 'municipality_code',
        'code_bureau_vote': 'polling_station_code',
        'geometry': 'geometry'
    }
    data = data[column_names.keys()].copy()
    data.rename(columns=column_names, inplace=True)
    return data


def format_polling_station_codes(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data['polling_station_code_is_correct'] = data['polling_station_code'].apply(lambda x: is_castable_to_int(x))
    data = data.loc[data['polling_station_code_is_correct']].copy()
    data.drop(columns=['polling_station_code_is_correct'], inplace=True)
    data['polling_station_code'] = data['polling_station_code'].apply(lambda x: str(x).zfill(4))
    return data


def convert_municipality_codes_to_parent_municipality_codes(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    crosswalk_parent_municipality_code_to_municipality_code = get_crosswalk_parent_municipality_code_to_municipality_code()
    data['parent_municipality_code'] = data['municipality_code'].apply(lambda x: crosswalk_parent_municipality_code_to_municipality_code[x] if x in crosswalk_parent_municipality_code_to_municipality_code else pd.NA)
    data.drop(columns=['municipality_code'], inplace=True)
    data.dropna(subset=['parent_municipality_code'], inplace=True)
    return data


def get_crosswalk_parent_municipality_code_to_municipality_code() -> Dict[str, str]:
    crosswalk = pd.read_csv(config.get_crosswalk_parent_municipality_code_to_municipality_code_file_path(), low_memory=False, dtype={'com': str, 'comparent': str})[['com', 'comparent']]
    crosswalk.rename(columns={'com': 'municipality_code', 'comparent': 'parent_municipality_code'}, inplace=True)
    crosswalk.fillna(axis=1, inplace=True, method='ffill')
    crosswalk_map = crosswalk.set_index('municipality_code')['parent_municipality_code'].to_dict()
    return crosswalk_map


def set_polling_station_code(data):
    data['polling_station'] = data['parent_municipality_code'] + data['polling_station_code']
    return data


def drop_redundant_geo_information(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns=['polling_station_code', 'parent_municipality_code'], inplace=True)
    return data


def is_castable_to_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False