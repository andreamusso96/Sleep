from typing import List

import pandas as pd

from . data import data


def get_geo_data(iris: List[str] = None) -> pd.DataFrame:
    iris_ = iris if iris is not None else data.index
    return data.data.loc[iris_].copy()