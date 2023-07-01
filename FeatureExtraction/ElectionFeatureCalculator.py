from enum import Enum
from typing import List

import numpy as np
import pandas as pd

from DataInterface.ElectionDataInterface import ElectionData
from DataInterface.GeoDataInterface import GeoData, GeoDataType
from FeatureExtraction.Feature import Feature
from FeatureExtraction.FeatureCalculator import FeatureCalculator


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


class ElectionFeatureCalculator(FeatureCalculator):
    def __init__(self, election_data: ElectionData, geo_data: GeoData):
        super().__init__()
        self.election_data = election_data
        self.geo_data = geo_data

    def get_election_feature(self, feature: ElectionFeatureName, subset: List[str] = None) -> Feature:
        if feature == ElectionFeatureName.ENTROPY:
            election_result = self.get_election_result_by_iris(subset=subset)
            entropy_vals = election_result.apply(self.entropy, axis=1).to_frame(name=ElectionFeatureName.ENTROPY.value)
            entropy = Feature(data=entropy_vals, name=ElectionFeatureName.ENTROPY.value)
            return entropy
        elif feature == ElectionFeatureName.SIMPSON:
            election_result = self.get_election_result_by_iris(subset=subset)
            simpson_vals = election_result.apply(self.simpson, axis=1).to_frame(name=ElectionFeatureName.SIMPSON.value)
            simpson = Feature(data=simpson_vals, name=ElectionFeatureName.SIMPSON.value)
            return simpson
        elif feature == ElectionFeatureName.TURNOUT:
            return self.get_turnout_by_iris(subset=subset)
        elif feature == ElectionFeatureName.POLARIZATION:
            return self.get_polarization_by_iris(subset=subset)
        else:
            raise NotImplementedError

    def get_election_result_by_iris(self, subset: List[str]) -> pd.DataFrame:
        table_results_polling_station_by_list_num = self.election_data.get_election_data_table(column='list_number', value='pct_votes_to_list_among_votes')
        table_results_iris_by_list_num = self._aggregate_to_iris_level(polling_station_level_data=table_results_polling_station_by_list_num, subset=subset, agg_function='mean')
        return table_results_iris_by_list_num

    def get_turnout_by_iris(self, subset: List[str]) -> Feature:
        abst = self.election_data.get_polling_station_metadata(column='pct_abstentions') / 100
        turnout = (1 - abst).rename(columns={'pct_abstentions': 'turnout'})
        turnout_by_iris = self._aggregate_to_iris_level(polling_station_level_data=turnout, subset=subset, agg_function='mean')
        turnout = Feature(data=turnout_by_iris, name=ElectionFeatureName.TURNOUT.value)
        return turnout

    def get_polarization_by_iris(self, subset: List[str]) -> Feature:
        election_result_distribution_by_polling_station = self.election_data.get_election_data_table(column='list_number', value='pct_votes_to_list_among_votes') / 100
        party_positions = self.election_data.party_orientation.set_index('list_number')[['left_right_position']]
        polarization_by_polling_station = election_result_distribution_by_polling_station.apply(lambda x: ElectionFeatureCalculator._weighted_std(values=party_positions.values.flatten(), weights=x.values.flatten()), axis=1).to_frame(name=ElectionFeatureName.POLARIZATION.value)
        polarization_by_iris = self._aggregate_to_iris_level(polling_station_level_data=polarization_by_polling_station, subset=subset, agg_function='mean')
        polarization = Feature(data=polarization_by_iris, name=ElectionFeatureName.POLARIZATION.value)
        return polarization

    def get_votes_for_party_by_iris(self, subset: List[str], party: Party) -> Feature:
        votes_for_party = self.election_data.get_votes_for_list(list_number=party.value) / 100
        votes_for_party_by_iris = self._aggregate_to_iris_level(polling_station_level_data=votes_for_party, subset=subset, agg_function='mean')
        votes_for_party = Feature(data=votes_for_party_by_iris.rename(columns={'pct_votes_to_list_among_votes': self.election_data.get_list_name(list_number=party.value)}), name=self.election_data.get_list_name(list_number=party.value))
        return votes_for_party

    def _aggregate_to_iris_level(self, polling_station_level_data: pd.DataFrame, subset: List[str], agg_function: str):
        polling_station_iris_map = self._get_polling_station_iris_map(subset=subset)
        polling_station_level_data_with_iris_info = polling_station_level_data.merge(polling_station_iris_map, left_index=True, right_index=True, how='inner')
        iris_level_data = polling_station_level_data_with_iris_info.groupby(by=GeoDataType.IRIS.value).agg(agg_function)
        return iris_level_data

    def _get_polling_station_iris_map(self, subset: List[str]):
        polling_station_iris_map = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset, other_geo_data_types=GeoDataType.POLLING_STATION).set_index(GeoDataType.POLLING_STATION.value)[GeoDataType.IRIS.value]
        return polling_station_iris_map

    @staticmethod
    def _weighted_std(values: np.ndarray, weights: np.ndarray):
        if np.sum(weights) == 0:
            return 0
        else:
            average = np.average(values, weights=weights)
            variance = np.average((values - average) ** 2, weights=weights)
            return np.sqrt(variance)



