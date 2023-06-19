from typing import Union, List

from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class Regression:
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.labels = labels
        self.scaler = StandardScaler()
        self.features_scaled = pd.DataFrame(self.scaler.fit_transform(self.features), columns=self.features.columns, index=self.features.index)
        self.label_vals = self.labels.values.flatten()
        self.reg = self._run_regression(features=self.features_scaled, labels=self.label_vals)

    def plot(self, x_axis: str, title: str = 'Regression', color: Union[str, pd.DataFrame] = None, ytickvals: np.ndarray = None, yticktext: List[str] = None):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Correlation', 'Regression Table'), specs=[[{"type": "scatter"}, {"type": "table"}]], column_widths=[0.7, 0.3])
        trace_data, trace_regression_forecasts = self._plot_correlation_plot(x_axis=x_axis, color=color)
        trace_table = self._plot_table()

        fig.add_trace(trace_data, row=1, col=1)
        fig.add_trace(trace_regression_forecasts, row=1, col=1)
        fig.add_trace(trace_table, row=1, col=2)
        self._layout(fig=fig, x_axis_label=x_axis, ytickvals=ytickvals, yticktext=yticktext, title=title)
        fig.show(renderer='browser')

    def _plot_table(self):
        headers, rows = self._prepare_table()
        column_widths = np.ones(len(headers))
        column_widths[0] = 2
        trace_table = go.Table(header=dict(values=headers), cells=dict(values=[x for x in zip(*rows)]), columnwidth=column_widths)
        return trace_table

    def _plot_correlation_plot(self, x_axis: str, color: Union[str, pd.DataFrame]):
        sorted_features, sorted_labels, sorted_color = self._sort_features(sort_by=x_axis, color=color)
        x_vals = sorted_features[x_axis].values
        trace_data = go.Scatter(x=x_vals, y=sorted_labels, mode='markers', name='Data', marker=dict(color=sorted_color), text=sorted_features.index)
        trace_regression_forecasts = go.Scatter(x=x_vals, y=self.reg.params[x_axis] * x_vals + self.reg.params['const'], mode='markers', marker=dict(symbol='x'), name='Regression')
        return trace_data, trace_regression_forecasts

    def _layout(self, fig: go.Figure,  x_axis_label: str, ytickvals: np.ndarray, yticktext: List[str], title: str):
        fig.update_layout(title=title, xaxis_title=x_axis_label, yaxis_title='Expected bed time', template='plotly')
        if ytickvals is not None and yticktext is not None:
            fig.update_yaxes(tickvals=ytickvals, ticktext=yticktext)

    def _sort_features(self, sort_by: str, color: Union[str, pd.DataFrame, None]):
        sort_index = np.argsort(self.features_scaled[sort_by].values)
        sorted_features = self.features_scaled.iloc[sort_index]
        sorted_labels = self.label_vals[sort_index]
        sorted_color = self._get_color(color=color, sorted_features=sorted_features, sort_index=sort_index)
        return sorted_features, sorted_labels, sorted_color

    def _get_color(self, color: Union[str, pd.DataFrame, None], sorted_features, sort_index):
        if isinstance(color, pd.DataFrame):
            merged = pd.merge(self.features, color, left_index=True, right_index=True, how='left', suffixes=('_feature', ''))
            merged = merged.iloc[sort_index]
            return merged[color.columns[0]].values
        elif isinstance(color, str):
            return sorted_features[color].values
        else:
            return 'blue'

    def _prepare_table(self):
        headers = ['Variable', 'Coefficient', 'P-value', 'T-value']
        rows = []
        for i in range(len(self.reg.params)):
            var_name = self.reg.params.index[i]
            rows.append([f"{var_name}", Regression._format_float(x=self.reg.params[i]), Regression._format_float(x=self.reg.pvalues[i]), Regression._format_float(x=self.reg.tvalues[i])])

        return headers, rows

    @staticmethod
    def _format_float(x):
        return '%s' % float('%.3g' % x)

    @staticmethod
    def _run_regression(features: pd.DataFrame, labels: np.ndarray):
        features = add_constant(features)
        ols = OLS(endog=labels, exog=features, hasconst=True)
        results = ols.fit()
        return results