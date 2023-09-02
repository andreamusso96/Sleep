from enum import Enum

from .data import data


class Party(Enum):
    FRANCE_INSOUMISE = 1
    RENAISSANCE = 5
    ECOLOGIE = 30
    LEPEN = 23


def get_votes_for_party(party: Party):
    votes = data.data[data.data['list_number'] == party.value][['pct_votes_to_list_among_votes']]
    return votes