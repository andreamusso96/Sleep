from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import mobile_traffic as mt
import insee
import noise


def night_screen_index_insee_tile(screen_time_data: Dict[mt.City, xr.DataArray]) -> pd.DataFrame:
    night_screen_index = []
    for city in screen_time_data:
        night_screen_index_city = night_screen_index_insee_tile_city(traffic_data=screen_time_data[city])
        night_screen_index_city['city'] = city.value
        night_screen_index.append(night_screen_index_city)

    night_screen_index = pd.concat(night_screen_index)
    return night_screen_index


def night_screen_index_insee_tile_city(traffic_data: xr.DataArray) -> pd.DataFrame:
    traffic_data_location_by_time = traffic_data.sum(dim='service')
    total_traffic_location = traffic_data_location_by_time.sum(dim='time')
    traffic_probability_location_by_time = (traffic_data_location_by_time / total_traffic_location).to_pandas()
    cumulative_traffic_distribution_location_by_time = traffic_probability_location_by_time.cumsum(axis=1)
    mean_cumulative_traffic_distribution_location_by_time = cumulative_traffic_distribution_location_by_time.mean(axis=0)
    deviation_from_mean = -1 * cumulative_traffic_distribution_location_by_time.subtract(mean_cumulative_traffic_distribution_location_by_time, axis=1)
    night_screen_index = deviation_from_mean.sum(axis=1).to_frame(name='night_screen_index')
    return night_screen_index


def log2_mean_income_insee_tile(insee_tiles: List[str]) -> pd.DataFrame:
    income = insee.tile.get_data(tile=insee_tiles, var_name='Ind_snv', shares=True)
    income = np.log2(income)
    income = income.rename(columns={'Ind_snv': 'log2_income'})
    return income


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


def log2_amenity_counts_insee_tile(insee_tiles: List[str], buffer_size_m: float = 1000, frequently_visited_amenities: bool = False) -> pd.DataFrame:
    insee_tiles_geo = insee.tile.get_geo_data(tile=insee_tiles)
    insee_tiles_buffered = insee_tiles_geo.buffer(buffer_size_m)
    insee_tiles_buffered = gpd.GeoDataFrame(geometry=insee_tiles_buffered, index=insee_tiles_buffered.index)
    amenity_counts = insee.equipment.get_equipment_counts(polygons=insee_tiles_buffered, resolution=insee.equipment.EquipmentResolution.HIGH)

    if frequently_visited_amenities:
        frequently_visited_amenities = get_frequently_visited_amenities()
        amenity_counts = amenity_counts[frequently_visited_amenities].copy()

    col_name = 'log2_fv_amenity_counts' if frequently_visited_amenities else 'log2_amenity_counts'
    amenity_counts = amenity_counts.sum(axis=1).to_frame(name=col_name)
    amenity_counts = np.log2(1 + amenity_counts)
    return amenity_counts


def noise_levels_insee_tile(insee_tiles: List[str]):
    insee_tiles_geo = insee.tile.get_geo_data(tile=insee_tiles)
    noise_estimates = noise.get_noise_estimate(polygons=insee_tiles_geo, measurement=noise.Measurement.NIGHT)
    noise_estimates = noise_estimates[['noise_estimate']].copy()
    return noise_estimates


if __name__ == '__main__':
    from synthetic_data import load_synthetic_dataset
    fp = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/SyntheticData'
    d = load_synthetic_dataset(folder_path=fp, insee_tiles=True)
    nsi = night_screen_index_insee_tile(screen_time_data=d)
    i_tiles = list(nsi.index)
    # log2_income = log2_mean_income_insee_tile(insee_tiles=i_tiles)
    # log2_fv_amenities = log2_amenity_counts_insee_tile(insee_tiles=i_tiles, frequently_visited_amenities=True)
    # log2_amenities = log2_amenity_counts_insee_tile(insee_tiles=i_tiles, frequently_visited_amenities=False)
    i_tiles_paris = nsi.loc[nsi['city'] == 'Paris'].index
    noise_levels = noise_levels_insee_tile(insee_tiles=i_tiles_paris)

