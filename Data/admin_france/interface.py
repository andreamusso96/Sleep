from typing import List, Union

import pandas as pd

from . import data


def get_admin_data(iris: Union[str, List[str]] = None, var_name: Union[str, List[str]] = None) -> pd.DataFrame:
    iris_ = process_union_input(iris)
    var_name_ = process_union_input(var_name)
    iris_ = iris_ if iris_ is not None else data.data.index
    var_name_ = var_name_ if var_name_ is not None else data.data.columns
    return data.dataloc[iris_, var_name_].copy()


def process_union_input(i: Union[str, List[str]]) -> List[str]:
    if isinstance(i, str):
        return [i]
    else:
        return i