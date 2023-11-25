from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import mobile_traffic as mt
import insee
import noise

from mobile_data import MobileData, ScreenTimeData


def night_screen_index_insee_tile(screen_time_data: ScreenTimeData) -> pd.DataFrame:
    night_screen_index = _night_screen_index(screen_time_data=screen_time_data, sum_over='service')
    return night_screen_index


def night_screen_index_service(screen_time_data: ScreenTimeData) -> pd.DataFrame:
    night_screen_index = _night_screen_index(screen_time_data=screen_time_data, sum_over='insee_tile')
    night_screen_index.reset_index(names=['service'], inplace=True)
    return night_screen_index


def _night_screen_index(screen_time_data: ScreenTimeData, sum_over: str) -> pd.DataFrame:
    night_screen_index = []
    for city in screen_time_data.cities():
        screen_time_data_city = screen_time_data.data[city]
        night_screen_index_city = _night_screen_index_city(screen_time_data=screen_time_data_city, sum_over=sum_over)
        night_screen_index_city['city'] = city.value
        night_screen_index.append(night_screen_index_city)

    night_screen_index = pd.concat(night_screen_index)
    return night_screen_index


def _night_screen_index_city(screen_time_data: xr.DataArray, sum_over: str):
    screen_time_data_x_by_time = screen_time_data.sum(dim=sum_over)
    total_screen_time_x = screen_time_data_x_by_time.sum(dim='time')
    screen_time_probability_x_by_time = (screen_time_data_x_by_time / total_screen_time_x).to_pandas()
    difference = _compute_difference_between_cumulative_distribution_and_mean_of_cumulative_distributions(probability_distributions=screen_time_probability_x_by_time)
    night_screen_index = -1 * difference.rename(columns={'difference': 'night_screen_index'})
    return night_screen_index


def _compute_difference_between_cumulative_distribution_and_mean_of_cumulative_distributions(probability_distributions: pd.DataFrame) -> pd.DataFrame:
    cumulative_distribution = probability_distributions.cumsum(axis=1)
    mean_cumulative_distribution = cumulative_distribution.mean(axis=0)
    difference = cumulative_distribution.subtract(mean_cumulative_distribution, axis=1)
    difference = difference.sum(axis=1).to_frame(name='difference')
    return difference


def log2_mean_income_insee_tile(insee_tiles: List[str]) -> pd.DataFrame:
    income = insee.tile.get_data(tile=insee_tiles, var_name='Ind_snv', shares=True)
    income = np.log2(income)
    income = income.rename(columns={'Ind_snv': 'log2_income'})
    return income


def map_insee_tile_to_income_category(insee_tiles: List[str], income_quantiles: List[float]) -> Dict[str, str]:
    log2_income = log2_mean_income_insee_tile(insee_tiles=insee_tiles)
    log2_income_quantiles = np.quantile(log2_income['log2_income'], q=income_quantiles)
    bins = [-np.inf] + log2_income_quantiles.tolist() + [np.inf]
    income_categories = pd.cut(log2_income['log2_income'], bins=bins, labels=[f'q{k}' for k in range(len(bins) - 1)])
    map_insee_tile_to_income_category = income_categories.to_dict()
    return map_insee_tile_to_income_category


def get_frequently_visited_amenities():
    frequently_visited_amenities = [
        "A203",  # BANQUE, CAISSE D’ÉPARGNE
        "A206",  # BUREAU DE POSTE
        "A207",  # RELAIS POSTE
        "A208",  # AGENCE POSTALE
        "A501",  # COIFFURE
        "A504",  # RESTAURANT- RESTAURATION RAPIDE
        "B101",  # HYPERMARCHÉ
        "B102",  # SUPERMARCHÉ
        "B201",  # SUPÉRETTE
        "B203",  # BOULANGERIE
        "B204",  # BOUCHERIE CHARCUTERIE
        "B301",  # LIBRAIRIE, PAPETERIE, JOURNAUX
        "B316",  # STATION SERVICE
        "C101",  # ÉCOLE MATERNELLE
        "C104",  # ÉCOLE ÉLÉMENTAIRE
        "C201",  # COLLÈGE
        "D201",  # MÉDECIN GÉNÉRALISTE
        "D307",  # PHARMACIE
        "E101",  # TAXI-VTC
        "E107",  # GARE DE VOYAGEURS D'INTERET NATIONAL
        "E108",  # GARE DE VOYAGEURS D'INTERET RÉGIONAL
        "F101",  # BASSIN DE NATATION
        "F103",  # TENNIS
        "F107",  # ATHLÉTISME
        "F121",  # SALLES MULTISPORTS (GYMNASES)
        "F303",  # CINÉMA
        "F312",  # EXPOSITION ET MEDIATION CULTURELLE
    ]
    return frequently_visited_amenities


def get_open_at_night_amenities():
    open_at_night_amenities = [
        "A101",  # POLICE
        "A104",  # GENDARMERIE
        "A504",  # RESTAURANT- RESTAURATION RAPIDE
        "B101",  # HYPERMARCHÉ
        "B316",  # STATION SERVICE
        "D106",  # URGENCE
        "D303",  # AMBULANCE
        "D307",  # PHARMACIE
        "E102",  # AÉROPORT
        "E107",  # GARE DE VOYAGEURS D'INTERET NATIONAL
        "E108",  # GARE DE VOYAGEURS D'INTERET RÉGIONAL
        "F101",  # BASSIN DE NATATION
        "F103",  # TENNIS
        "F107",  # ATHLÉTISME
        "F121",  # SALLES MULTISPORTS (GYMNASES)
        "F303",  # CINÉMA
        "G102",  # HÔTEL
        "G103",  # CAMPING
    ]

    return open_at_night_amenities


class AmenityType:
    ALL = 'all'
    FREQUENTLY_VISITED = 'frequently_visited'
    OPEN_AT_NIGHT = 'open_at_night'


def log2_amenity_counts_insee_tile(insee_tiles: List[str], buffer_size_m: float = 1000, amenity_type: AmenityType = AmenityType.ALL) -> pd.DataFrame:
    insee_tiles_geo = insee.tile.get_geo_data(tile=insee_tiles)
    insee_tiles_buffered = insee_tiles_geo.buffer(buffer_size_m)
    insee_tiles_buffered = gpd.GeoDataFrame(geometry=insee_tiles_buffered, index=insee_tiles_buffered.index)
    amenity_counts = insee.equipment.get_equipment_counts(polygons=insee_tiles_buffered, resolution=insee.equipment.EquipmentResolution.HIGH)

    col_name = 'log2_amenity_counts'
    amenities = amenity_counts.columns
    if amenity_type == AmenityType.FREQUENTLY_VISITED:
        frequently_visited_amenities = get_frequently_visited_amenities()
        amenity_counts = amenity_counts[np.intersect1d(amenities, frequently_visited_amenities)].copy()
        col_name = 'log2_fv_amenity_counts'
    elif amenity_type == AmenityType.OPEN_AT_NIGHT:
        open_at_night_amenities = get_open_at_night_amenities()
        amenity_counts = amenity_counts[np.intersect1d(amenities, open_at_night_amenities)].copy()
        col_name = 'log2_oan_amenity_counts'

    amenity_counts = amenity_counts.sum(axis=1).to_frame(name=col_name)
    amenity_counts = np.log2(1 + amenity_counts)
    return amenity_counts


def noise_levels_insee_tile(insee_tiles: List[str]):
    insee_tiles_geo = insee.tile.get_geo_data(tile=insee_tiles)
    noise_estimates = noise.get_noise_estimate(polygons=insee_tiles_geo, measurement=noise.Measurement.NIGHT)
    noise_estimates = noise_estimates[['noise_estimate']].copy()
    return noise_estimates


if __name__ == '__main__':
    from mobile_data import TrafficData
    td = TrafficData.load_dataset(synthetic=False, insee_tiles=True)
    size_bytes = 0
    for city in td.cities():
        data_city = td.data[city]
        size_bytes += data_city.nbytes

    size_gb = size_bytes / 1e9
    print('Size in GB:', size_gb)

