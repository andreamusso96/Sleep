from typing import List
from enum import Enum

import pandas as pd

from DataPreprocessing.ElectionData.Data import ElectionDataComplete
from DataInterface.DataInterface import DataInterface
from DataInterface.GeoDataInterface import GeoData, GeoDataType


class ElectionFeatureName(Enum):
    ENTROPY = 'entropy'
    SIMPSON = 'simpson'
    TURNOUT = 'turnout'
    POLARIZATION = 'polarization'
    PARTY_VOTES = 'party_votes'


class Party(Enum):
    FRANCE_INSOUMISE = 1
    RENAISSANCE = 5
    ECOLOGIE = 30
    LEPEN = 23


class ElectionData(DataInterface):
    def __init__(self):
        super().__init__()
        self._election_data_complete = ElectionDataComplete()
        self.data = self._election_data_complete.data
        self.party_orientation = self._election_data_complete.party_orientation.data
        self.list_number_to_label_map = self._get_list_number_to_list_label_map()

    def get_election_data_table(self, column: str, value: str):
        table_value_polling_station_by_column = self.data.pivot(index=GeoDataType.POLLING_STATION.value, columns=column, values=value)
        return table_value_polling_station_by_column

    def get_polling_station_metadata(self, column: str):
        metadata = self.data[[GeoDataType.POLLING_STATION.value, column]].drop_duplicates(subset=[GeoDataType.POLLING_STATION.value]).set_index(GeoDataType.POLLING_STATION.value)
        return metadata

    def get_votes_for_list(self, list_number: int):
        votes_for_list = self.data[self.data['list_number'] == list_number][[GeoDataType.POLLING_STATION.value, 'pct_votes_to_list_among_votes']].set_index(GeoDataType.POLLING_STATION.value)
        return votes_for_list

    def get_list_name(self, list_number: int):
        return self.list_number_to_label_map[list_number]

    def _get_list_number_to_list_label_map(self):
        list_number_to_list_label_map = self.data[['list_number', 'short_list_label']].drop_duplicates(subset=['list_number', 'short_list_label']).set_index('list_number').to_dict()['short_list_label']
        return list_number_to_list_label_map