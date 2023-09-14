from typing import List, Union

import numpy as np
import pandas as pd

from . data import data


def get_data(iris: Union[str, List[str]] = None, var_name: Union[str, List[str]] = None, shares: bool = False) -> pd.DataFrame:
    iris_ = process_union_input(iris)
    var_name_ = process_union_input(var_name)
    iris_ = np.intersect1d(iris_, data.data.index) if iris_ is not None else data.data.index
    var_name_ = var_name_ if var_name_ is not None else data.data.columns

    data_ = data.data.loc[iris_, var_name_].copy()
    if shares:
        pop = data.data.loc[iris_, 'P19_POP']
        data_ = data_.div(pop, axis=0)
    return data_


def process_union_input(i: Union[str, List[str]]) -> List[str]:
    if isinstance(i, str):
        return [i]
    else:
        return i