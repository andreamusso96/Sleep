import pandas as pd


from .data import data


def get_geo_data() -> pd.DataFrame:
    return data.data