from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import statsmodels.regression.linear_model as statmodels_linear_model

from . import utils


def regression(data: pd.DataFrame, x_axis: str, y_axis: str, outlier_threshold: float = 0.99, marker_size: float = None, marker_opacity: float = 1, marker_color: str = 'black', line_color: str = 'red', line_width: float = None, showlegend: bool = False) -> Tuple[go.Scatter, go.Scatter, statmodels_linear_model.RegressionResults]:
    data_ = data.copy()

    data_ = utils.remove_outliers(data=data_, column=x_axis, threshold=outlier_threshold)
    data_ = utils.remove_outliers(data=data_, column=y_axis, threshold=outlier_threshold)

    reg = utils.run_regression(data=data_, x_axis=x_axis, y_axis=y_axis)
    intercept, slope = reg.params[0], reg.params[1]

    trace_scatter = go.Scatter(x=data_[x_axis], y=data_[y_axis], mode='markers', marker=dict(opacity=marker_opacity, size=marker_size, color=marker_color), showlegend=showlegend)
    trace_regression_line = go.Scatter(x=data_[x_axis], y=intercept + slope * data_[x_axis], mode='lines', line=dict(color=line_color, width=line_width), showlegend=showlegend)
    return trace_scatter, trace_regression_line, reg