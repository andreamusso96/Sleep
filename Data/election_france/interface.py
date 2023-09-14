from enum import Enum
from typing import Union, List

import numpy as np
import pandas as pd

from .data import data


class Party(Enum):
    FRANCE_INSOUMISE = 1
    RENAISSANCE = 5
    ECOLOGIE = 30
    LEPEN = 23

    @staticmethod
    def get_extremist_party_list_number() -> List[int]:
        return [1, 13, 15, 19, 23]


def get_percentage_votes_for_party_at_iris_level(list_number: Union[int, List[int]], iris: List[str] = None, return_radius: bool = True) -> pd.DataFrame:
    list_number_ = list_number if isinstance(list_number, list) else [list_number]
    votes_at_polling_station_level = data.data.loc[data.data['list_number'].isin(list_number_)][['polling_station', 'list_number', 'pct_votes_to_list_among_votes', 'expressed_votes']].copy()
    votes_at_iris_level = _aggregate_votes_for_party_at_iris_level(votes_at_polling_station_level=votes_at_polling_station_level)
    radius = votes_at_iris_level[['iris', 'radius']].drop_duplicates().set_index('iris')
    votes_at_iris_level = votes_at_iris_level.pivot(index='iris', columns='list_number', values='pct_votes_to_list_among_votes')
    if return_radius:
        votes_at_iris_level = votes_at_iris_level.join(radius, how='left')

    iris_ = np.intersect1d(iris, votes_at_iris_level.index) if iris is not None else votes_at_iris_level.index
    return votes_at_iris_level.loc[iris_].copy()


def _aggregate_votes_for_party_at_iris_level(votes_at_polling_station_level: pd.DataFrame) -> pd.DataFrame:
    votes_at_polling_station_level = _add_iris_label_to_polling_station_level_data(polling_station_level_data=votes_at_polling_station_level)
    votes_at_polling_station_level.dropna(subset=['iris'], inplace=True)
    votes_at_polling_station_level['fraction_votes_to_list_among_votes'] = votes_at_polling_station_level['pct_votes_to_list_among_votes'] / 100
    votes_at_polling_station_level['number_votes_to_list'] = votes_at_polling_station_level['fraction_votes_to_list_among_votes'] * votes_at_polling_station_level['expressed_votes']
    votes_at_iris_level = votes_at_polling_station_level.groupby(['iris', 'list_number']).agg({'number_votes_to_list': 'sum', 'expressed_votes': 'sum', 'radius': 'median'})
    votes_at_iris_level['pct_votes_to_list_among_votes'] = 100 * votes_at_iris_level['number_votes_to_list'] / votes_at_iris_level['expressed_votes']
    votes_at_iris_level.reset_index(inplace=True)
    return votes_at_iris_level


def _add_iris_label_to_polling_station_level_data(polling_station_level_data: pd.DataFrame) -> pd.DataFrame:
    matching_iris_polling_station = data.matching_iris_polling_station.set_index('polling_station')
    polling_station_level_data.set_index('polling_station', inplace=True)
    polling_station_level_data_with_iris_labels = polling_station_level_data.join(matching_iris_polling_station, how='left')
    return polling_station_level_data_with_iris_labels