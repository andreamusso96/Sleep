from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from . import utils


def plot(data: gpd.GeoDataFrame, column: str, ax: plt.Axes, log: bool = False, outlier_threshold: float = 0.99, legend: bool = True, cmap: str = 'Greys', **kwargs):
    data_ = data.copy()
    if log:
        data_ = utils.log_transform_column(data=data_, column=column)

    data_ = utils.remove_outliers(data=data_, column=column, threshold=outlier_threshold)
    data_ = utils.scale_column(data=data_, column=column)
    data_.plot(column=column, ax=ax, legend=legend, cmap=cmap, **kwargs)
    ax.set_axis_off()
    return ax


def _scale_data(data: gpd.GeoDataFrame, column: str):
    data[column] = StandardScaler().fit_transform(data[column].values.reshape(-1, 1)).reshape(-1)
    return data
