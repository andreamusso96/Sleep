from typing import List, Tuple, Dict
import itertools

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from . import utils


def barchart_coefficient_with_all_control_combinations(data: pd.DataFrame, x_axis: str, y_axis: str, controls: Dict[str, List[str]], control_labels: Dict[str, str]) -> go.Bar:
    data_ = data.copy()
    regression_results = _run_regression_with_all_control_combinations(data=data_, x_axis=x_axis, y_axis=y_axis, controls=controls, control_labels=control_labels)
    barchart_trace = go.Bar(x=regression_results['controls'], y=regression_results['coefficient'], error_y=dict(type='data', array=regression_results['std_err']))
    return barchart_trace


def _run_regression_with_all_control_combinations(data: pd.DataFrame, x_axis: str, y_axis: str, controls: Dict[str, List[str]], control_labels: Dict[str, str]) -> pd.DataFrame:
    control_combinations = [[]]
    for i in range(1, len(controls) + 1):
        control_combinations += list(itertools.combinations(controls.keys(), i))

    regression_results = []
    for control_combination in control_combinations:
        control_combination_vars = []
        for control in control_combination:
            control_combination_vars += controls[control]
        control_combination_label = '+'.join([control_labels[control] for control in control_combination]) if len(control_combination) > 0 else 'No'
        reg = utils.run_multivariable_regression(data=data[[y_axis, x_axis] + control_combination_vars], y_axis=y_axis)
        coefficient, std_err = reg.params[x_axis], reg.bse[x_axis]
        regression_results.append([control_combination_label, coefficient, std_err])

    regression_results = pd.DataFrame(regression_results, columns=['controls', 'coefficient', 'std_err'])
    regression_results.sort_values(by='coefficient', inplace=True)
    return regression_results


def barchart_mean_values_along_axis(data: pd.DataFrame, nbins: int, x_axis: str, col_label: Dict[str, str], outlier_threshold=0.99) -> List[go.Bar]:
    data_ = data.copy()
    data_ = utils.remove_outliers(data=data_, column=x_axis, threshold=outlier_threshold)
    scaler = StandardScaler()
    data_ = pd.DataFrame(scaler.fit_transform(data_), columns=data_.columns, index=data_.index)
    grouped_data_mean, grouped_data_std, grouped_data_nobs = _group_data(data=data_, nbins=nbins, x_axis=x_axis)
    traces = []
    for column in grouped_data_mean.columns:
        conf_interval = 1.96 * np.power(grouped_data_std[column] / grouped_data_nobs[column], 0.5)
        trace_col = go.Bar(x=grouped_data_mean.index, y=grouped_data_mean[column], error_y=dict(type='data', array=conf_interval, visible=True), name=col_label[column])
        traces.append(trace_col)
    return traces


def _group_data(data: pd.DataFrame, nbins: int, x_axis: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data['bin'] = pd.cut(data[x_axis], bins=nbins)
    data['bin'] = data['bin'].apply(lambda x: x.mid)
    data = data.drop(columns=[x_axis])
    grouped_data = data.groupby(by='bin')
    grouped_data_mean = grouped_data.mean()
    grouped_data_std = grouped_data.std()
    grouped_data_nobs = grouped_data.count()
    return grouped_data_mean, grouped_data_std, grouped_data_nobs