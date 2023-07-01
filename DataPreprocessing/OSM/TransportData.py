from enum import Enum
from typing import List
from unidecode import unidecode

import osmnx as ox
import pandas as pd
import geopandas as gpd
from Utils import City


class TransportVector(Enum):
    TRAIN = 'train'
    BUS = 'bus'
    TRAM = 'tram'
    SUBWAY = 'subway'
    FERRY = 'ferry'
    FUNICULAR = 'funicular'


class TransportDataDownloader:
    def __init__(self, city: City, city_geometry: gpd.GeoDataFrame):
        self.city = city
        self._city_geometry = city_geometry['geometry'].iloc[0]

    def download(self, transport_vectors: List[TransportVector] = None):
        if transport_vectors is None:
            transport_vectors = [TransportVector.TRAIN, TransportVector.BUS, TransportVector.TRAM, TransportVector.SUBWAY, TransportVector.FERRY, TransportVector.FUNICULAR]

        public_transport_stop_positions = ox.geometries_from_polygon(polygon=self._city_geometry, tags={'public_transport': 'stop_position'})
        transport_vectors_in_city = list(set(public_transport_stop_positions.columns).intersection(set([transport_vector.value for transport_vector in transport_vectors])))
        public_transport_stop_positions = public_transport_stop_positions[transport_vectors_in_city + ['name', 'geometry']]

        clean_stop_positions = []
        for transport_vector in transport_vectors_in_city:
            clean_stop_positions.append(self._clean_data(public_transport_stop_positions=public_transport_stop_positions, transport_vector=transport_vector))

        public_transport_stop_positions = pd.concat(clean_stop_positions, axis=0, ignore_index=True)
        return public_transport_stop_positions

    def _city_geometry(self):
        pass

    @staticmethod
    def _clean_data(public_transport_stop_positions: gpd.GeoDataFrame, transport_vector: str):
        mask = public_transport_stop_positions[transport_vector].notna()
        public_transport_stop_positions_vector = public_transport_stop_positions[mask].copy()
        public_transport_stop_positions_vector = TransportDataDownloader._reformat_names(public_transport_stop_positions=public_transport_stop_positions_vector)
        public_transport_stop_positions_vector = TransportDataDownloader._remove_doubles(public_transport_stop_positions=public_transport_stop_positions_vector)
        public_transport_stop_positions_vector = TransportDataDownloader._reformat_columns(public_transport_stop_positions=public_transport_stop_positions_vector, transport_vector=transport_vector)
        return public_transport_stop_positions_vector.reset_index(drop=True)

    @staticmethod
    def _remove_doubles(public_transport_stop_positions: gpd.GeoDataFrame):
        public_transport_stop_positions_without_doubles = public_transport_stop_positions.drop_duplicates(subset=['name'])
        return public_transport_stop_positions_without_doubles

    @staticmethod
    def _reformat_names(public_transport_stop_positions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        def reformat_string(string: str):
            string = string.lower()
            string = unidecode(string)
            string = string.replace(' ', '_')
            string = string.replace('-', '_')
            string = string.replace('___', '_')
            string = string.replace('__', '_')
            string = string.replace("'", '')
            return string


        public_transport_stop_positions['name'] = public_transport_stop_positions['name'].apply(lambda x: reformat_string(str(x)))
        return public_transport_stop_positions

    @staticmethod
    def _reformat_columns(public_transport_stop_positions: gpd.GeoDataFrame, transport_vector: str) -> gpd.GeoDataFrame:
        public_transport_stop_positions = public_transport_stop_positions[[transport_vector, 'name', 'geometry']]
        public_transport_stop_positions = public_transport_stop_positions.rename(columns={transport_vector: 'vector'})
        public_transport_stop_positions['vector'] = transport_vector
        return public_transport_stop_positions