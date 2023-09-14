from typing import Union

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.regression.linear_model as statmodels_linear_model


def remove_outliers(data: Union[pd.DataFrame, gpd.GeoDataFrame], column: str, threshold: float = 0.95) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    mask_outliers = (data[column] > data[column].quantile(threshold)) | (data[column] < data[column].quantile(1 - threshold))
    data = data.loc[~mask_outliers]
    return data


def log_transform_column(data: Union[pd.DataFrame, gpd.GeoDataFrame], column: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    data[column] = data[column].apply(lambda x: np.log(x + 1))
    return data


def scale_column(data: Union[pd.DataFrame, gpd.GeoDataFrame], column: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    data[column] = StandardScaler().fit_transform(data[column].values.reshape(-1, 1)).reshape(-1)
    return data


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)
    return data


def run_regression(data: pd.DataFrame, x_axis: str, y_axis: str) -> statmodels_linear_model.RegressionResults:
    y = data[y_axis]
    X = data[x_axis]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results


def run_multivariable_regression(data: pd.DataFrame, y_axis: str) -> statmodels_linear_model.RegressionResults:
    y = data[y_axis]
    X = data.drop(columns=[y_axis])
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results