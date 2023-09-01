import pandas as pd

from . import config


def save_preprocessed_admin_data():
    admin_data = generate_admin_data()
    admin_data.to_csv(config.get_data_file_path(), index=True)


def save_preprocessed_admin_metadata():
    admin_metadata = generate_admin_metadata()
    admin_metadata.to_csv(config.get_metadata_file_path(), index=False)


def generate_admin_data() -> pd.DataFrame:
    equipment_counts = load_insee_data_file(file_name=config.INSEEDataFileName.EQUIPMENT_COUNTS)
    equipment_counts = format_equipment_counts_data_for_merge(data=equipment_counts)
    data = merge_insee_datasets(dataset1=equipment_counts, dataset2=load_insee_data_file(file_name=config.INSEEDataFileName.OCCUPATION))
    data = merge_insee_datasets(dataset1=data, dataset2=load_insee_data_file(file_name=config.INSEEDataFileName.COUPLES_FAMILIES_HOUSEHOLDS))
    data = merge_insee_datasets(dataset1=data, dataset2=load_insee_data_file(file_name=config.INSEEDataFileName.EDUCATION))
    data = merge_insee_datasets(dataset1=data, dataset2=load_insee_data_file(file_name=config.INSEEDataFileName.HOUSING))
    data = merge_insee_datasets(dataset1=data, dataset2=load_insee_data_file(file_name=config.INSEEDataFileName.INCOME))
    data.sort_index(inplace=True)
    return data


def merge_insee_datasets(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> pd.DataFrame:
    merged_datasets = dataset1.merge(dataset2, left_index=True, right_index=True, how='outer', suffixes=('', f'_duplicate'))
    merged_datasets.drop(columns=[c for c in merged_datasets.columns if c.endswith('_duplicate')], inplace=True)
    return merged_datasets


def format_equipment_counts_data_for_merge(data: pd.DataFrame) -> pd.DataFrame:
    data = data.pivot(index='iris', columns='TYPEQU', values='NB_EQUIP').fillna(0).reset_index()
    data.rename(columns={col: f'EQUIP_{col}' for col in data.columns if col != 'iris'}, inplace=True)
    data.set_index('iris', inplace=True)
    data.sort_index(inplace=True)
    return data


def generate_admin_metadata() -> pd.DataFrame:
    merged_metadata = pd.concat([load_insee_metadata_file(file_name=file_name) for file_name in config.INSEEDataFileName])
    merged_metadata = merged_metadata[~merged_metadata['COD_VAR'].duplicated()]
    merged_metadata = merged_metadata.sort_values(by='COD_VAR').reset_index(drop=True)
    return merged_metadata


def load_insee_data_file(file_name: config.INSEEDataFileName) -> pd.DataFrame:
    if file_name in [config.INSEEDataFileName.OCCUPATION, config.INSEEDataFileName.COUPLES_FAMILIES_HOUSEHOLDS, config.INSEEDataFileName.EDUCATION, config.INSEEDataFileName.HOUSING]:
        return preprocess_insee_data_file(file_name=file_name, year=2019)
    elif file_name == config.INSEEDataFileName.EQUIPMENT_COUNTS:
        return preprocess_equipment_counts_data_file(year=2021)
    elif file_name == config.INSEEDataFileName.INCOME:
        return preprocess_income_data_file(year=2019)
    else:
        raise NotImplementedError(f'File name {file_name} not implemented')


def load_insee_metadata_file(file_name: config.INSEEDataFileName) -> pd.DataFrame:
    if file_name != config.INSEEDataFileName.EQUIPMENT_COUNTS:
        return preprocess_insee_metadata_file(file_name=file_name, year=2019)
    else:
        return preprocess_equipment_counts_metadata_file(file_name=file_name, year=2021)


def preprocess_insee_data_file(file_name: config.INSEEDataFileName, year: int) -> pd.DataFrame:
    data = read_insee_data_file(file_name=file_name, year=year, sep=';', dtype={'IRIS': str}, low_memory=False)
    data.rename(columns={'IRIS': 'iris'}, inplace=True)
    data.drop(columns=['COM', 'LAB_IRIS', 'MODIF_IRIS'], inplace=True)
    data.sort_values(by='iris', inplace=True)
    data.set_index('iris', inplace=True)
    return data


def preprocess_equipment_counts_data_file(year: int) -> pd.DataFrame:
    columns_to_keep = ['DCIRIS', 'DOM', 'SDOM', 'TYPEQU', 'NB_EQUIP']
    data = read_insee_data_file(file_name=config.INSEEDataFileName.EQUIPMENT_COUNTS, year=year, sep=';', dtype={'DCIRIS': str}, usecols=columns_to_keep)
    data.rename(columns={'DCIRIS': 'iris'}, inplace=True)
    data = data.groupby(by=['iris', 'TYPEQU']).sum().reset_index()
    data.sort_values(by='iris', inplace=True)
    return data


def preprocess_income_data_file(year: int) -> pd.DataFrame:
    data = read_insee_data_file(file_name=config.INSEEDataFileName.INCOME, year=year, sep=',', dtype={'IRIS': str}, low_memory=False)
    data.rename(columns={'IRIS': 'iris'}, inplace=True)
    data['iris'] = data['iris'].apply(lambda x: str(x).zfill(9))
    data.sort_values(by='iris', inplace=True)
    data.set_index('iris', inplace=True)
    return data


def read_insee_data_file(file_name: config.INSEEDataFileName, year: int, **kwargs) -> pd.DataFrame:
    file_path = config.get_insee_data_file_path(file_name=file_name, year=year)
    data = pd.read_csv(file_path, **kwargs)
    return data


def preprocess_insee_metadata_file(file_name: config.INSEEDataFileName, year: int) -> pd.DataFrame:
    data = read_insee_metadata_file(file_name=file_name, year=year, sep=';')
    data = data[['COD_VAR', 'LIB_VAR_LONG']].copy()
    data.rename(columns={'LIB_VAR_LONG': 'DESC'}, inplace=True)
    data = data[~data['COD_VAR'].isin(['IRIS', 'COM', 'MODIF_IRIS', 'LAB_IRIS'])]
    return data


def preprocess_equipment_counts_metadata_file(file_name: config.INSEEDataFileName, year: int) -> pd.DataFrame:
    data = read_insee_metadata_file(file_name=file_name, year=year)
    data = data[['TYPEQU', 'TYPEQU_TITLE']].copy()
    data.rename(columns={'TYPEQU': 'COD_VAR', 'TYPEQU_TITLE': 'DESC'}, inplace=True)
    data['COD_VAR'] = data['COD_VAR'].apply(lambda x: f'EQUIP_{x}')
    return data


def read_insee_metadata_file(file_name: config.INSEEDataFileName, year: int, **kwargs) -> pd.DataFrame:
    file_path = config.get_insee_metadata_file_path(file_name=file_name, year=year)
    metadata = pd.read_csv(file_path, **kwargs)
    return metadata