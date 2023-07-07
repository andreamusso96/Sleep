from typing import Union, List, Dict, Callable
from itertools import accumulate
import operator

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import RegressionResults as RegressionResultsSM
from statsmodels.tools import add_constant
import plotly.express as px

from FeatureSelection.Controls import Control


class RegressionResult:
    def __init__(self, regression: RegressionResultsSM, treatment: str, outcome: str, control: Control, data: pd.DataFrame):
        self.effect = regression.params[treatment]
        self.const = regression.params['const']
        self.conf_high = regression.conf_int(alpha=0.05).loc[treatment].values[1]
        self.conf_low = regression.conf_int(alpha=0.05).loc[treatment].values[0]
        self.p_value = regression.pvalues[treatment]
        self._regression = regression
        self.treatment = treatment
        self.outcome = outcome
        self.control = control
        self.data = data

    def to_dataframe(self):
        return pd.DataFrame({'effect': self.effect, 'conf_high': self.conf_high, 'conf_low': self.conf_low, 'pvalue': self.p_value, 'treatment': self.treatment, 'control': self.control.name}, index=[0])


class RegressionPlotter:
    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, colors: Dict[str, str] = None):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.colors = colors if colors is not None else {}

    # FIGURES
    def scatter_plot_with_controls(self, regression_results: List[RegressionResult], axis_number: int, show: bool):
        trace_control_bar_chart = self.control_bar_chart_trace(regression_results=regression_results, axis_number=axis_number, show_controls_at_x=show)
        trace_scatter_data, trace_regression_line = self.scatter_plot_trace(regression_result=regression_results[0])

        if show:
            trace_table = self._table_trace(regression_results=regression_results)
            self._build_and_show_figure(trace1=trace_scatter_data, trace2=trace_regression_line, trace3=trace_control_bar_chart, trace4=trace_table, main_plot_title=f'Scatter plot of {self.treatment} and {self.outcome} with regression line')
        else:
            return trace_control_bar_chart, trace_scatter_data, trace_regression_line

    def heatmap_plot_with_controls(self, regression_results: List[RegressionResult], axis_number: int, bins: int, rescale_fct: Callable, show: bool):
        trace_control_bar_chart = self.control_bar_chart_trace(regression_results=regression_results, axis_number=axis_number, show_controls_at_x=show)
        trace_heat_map = self.heatmap_trace(bins=bins, rescale_fct=rescale_fct)
        trace_mean_outcome_line = self.mean_outcome_line_trace(bins=bins)

        if show:
            trace_table = self._table_trace(regression_results=regression_results)
            self._build_and_show_figure(trace1=trace_heat_map, trace2=trace_mean_outcome_line,
                                        trace3=trace_control_bar_chart, trace4=trace_table,
                                        main_plot_title=f'Heat map plot of {self.treatment} and {self.outcome} with mean line')
        else:
            return trace_control_bar_chart, trace_heat_map, trace_mean_outcome_line

    def _build_and_show_figure(self, trace1, trace2, trace3, trace4, main_plot_title: str):
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.03,
                            specs=[[{'rowspan': 2}, {}], [None, {'type': 'table'}]], subplot_titles=(main_plot_title, f'Effects of {self.treatment} on {self.outcome}', f'Dependent variable {self.outcome}'),
                            column_widths=[0.6, 0.4])
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=1)
        fig.add_trace(trace3, row=1, col=2)
        fig.add_trace(trace4, row=2, col=2)
        self._setup_layout(fig=fig)
        fig.show(renderer='browser')

    def _setup_layout(self, fig):
        fig.update_xaxes(title_text=self.treatment, showline=True, linecolor='black', linewidth=2, row=1, col=1)
        fig.update_yaxes(title_text=self.outcome, showline=True, linecolor='black', linewidth=2, row=1, col=1)
        fig.update_xaxes(title_text='Control', showline=True, linecolor='black', linewidth=2, row=1, col=2)
        fig.update_yaxes(title_text='Effect', showline=True, linecolor='black', linewidth=2, row=1, col=2)
        fig.update_layout(template='plotly_white', showlegend=False)

    # TRACES
    def control_bar_chart_trace(self, regression_results: List[RegressionResult], axis_number: int, show_controls_at_x: bool) -> go.Bar:
        effects = pd.concat([r.to_dataframe() for r in regression_results])
        controls = [r.control for r in regression_results]

        ys = effects['effect']
        xs = [c.name for c in controls] if show_controls_at_x else list(accumulate([' ' for _ in range(len(controls))], operator.add))
        error_y = dict(type='data', symmetric=False, array=effects['conf_high'] - effects['effect'], arrayminus=effects['effect'] - effects['conf_low'])
        colors_ = self._get_colors(controls=[r.control for r in regression_results])

        trace_bar_chart = go.Bar(x=xs, y=ys, error_y=error_y, xaxis=f'x{axis_number}', yaxis=f'y{axis_number}', marker=dict(color=colors_), showlegend=False)
        return trace_bar_chart

    def scatter_plot_trace(self, regression_result: RegressionResult):
        x_axis = regression_result.data[regression_result.treatment]
        y_axis = regression_result.data[regression_result.outcome]
        trace_scatter_data = go.Scatter(x=x_axis, y=y_axis, mode='markers', marker=dict(color='grey', opacity=0.3), showlegend=False)
        trace_regression_line = go.Scatter(x=x_axis, y=x_axis * regression_result.effect + regression_result.const, mode='lines', line=dict(color='black', width=2), showlegend=False)
        return trace_scatter_data, trace_regression_line

    def heatmap_trace(self, bins: int, rescale_fct: Callable):
        binned_data = self._bin_data(bins=bins)
        heatmap_data = binned_data.groupby([f'{self.treatment}_bin', f'{self.outcome}_bin']).size().reset_index(name='counts').pivot(columns=f'{self.treatment}_bin', index=f'{self.outcome}_bin', values='counts').fillna(0)
        heatmap = go.Heatmap(z=rescale_fct(heatmap_data.values), x=heatmap_data.columns, y=heatmap_data.index, colorscale='Greys', showscale=False)
        return heatmap

    def mean_outcome_line_trace(self, bins: int):
        binned_data = self._bin_data(bins=bins)
        outcome_grouped_by_treatment_bin = binned_data.groupby(f'{self.treatment}_bin')[self.outcome]
        mean = outcome_grouped_by_treatment_bin.mean()
        n_obs = outcome_grouped_by_treatment_bin.count()
        std = outcome_grouped_by_treatment_bin.std()
        conf_int = 1.96 * std / np.sqrt(n_obs)
        mask_enough_obs = n_obs > 5
        mean = mean.loc[mask_enough_obs]
        conf_int = conf_int.loc[mask_enough_obs]
        error_y = dict(type='data', array=conf_int, visible=True)
        trace_mean = go.Scatter(x=mean.index, y=mean.values, error_y=error_y, mode='lines', line=dict(color='rgb(255, 0, 255)', width=4), showlegend=False)
        return trace_mean

    def _bin_data(self, bins: int):
        data = self.data[[self.treatment, self.outcome]].copy()
        data[f'{self.treatment}_bin'] = pd.cut(data[self.treatment], bins=bins)
        data[f'{self.outcome}_bin'] = pd.cut(data[self.outcome], bins=bins)
        data[f'{self.treatment}_bin'] = data[f'{self.treatment}_bin'].apply(lambda x: x.mid)
        data[f'{self.outcome}_bin'] = data[f'{self.outcome}_bin'].apply(lambda x: x.mid)
        return data

    @staticmethod
    def _table_trace(regression_results: List[RegressionResult]):
        table = pd.concat([r.to_dataframe() for r in regression_results])[['controls', 'effect', 'pvalue']]
        column_widths = np.ones(len(table.columns))
        column_widths[0] = 2
        trace_table = go.Table(header=dict(values=table.columns), cells=dict(values=table.values.T), columnwidth=column_widths)
        return trace_table

    # UTILS

    def _get_colors(self, controls: List[Control]) -> List[str]:
        colors = []
        for control in controls:
            if control.name not in self.colors:
                n_color = len(self.colors)
                self.colors[control.name] = px.colors.qualitative.Plotly[n_color]

            colors.append(self.colors[control.name])
        return colors

    @staticmethod
    def _format_float(x):
        return '%s' % float('%.3g' % x)


class Regression:
    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str):
        self.data = data.sort_values(by=treatment)
        self.treatment = treatment
        self.outcome = outcome

    def run_regressions(self, control: Control):
        independent_vars = [self.treatment]
        independent_vars += control.var_names
        regression = self._ols_regression(independent_vars=independent_vars, dependent_var=self.outcome)
        result = RegressionResult(regression=regression, treatment=self.treatment, outcome=self.outcome, control=control, data=self.data)
        return result

    def _ols_regression(self, independent_vars, dependent_var):
        features = add_constant(self.data[independent_vars])
        ols = OLS(endog=self.data[dependent_var], exog=features, hasconst=True)
        results = ols.fit()
        return results

    def scatter_plot_with_controls(self, controls: List[Control], colors: Dict[str, str] = None, axis_number: int = 4, show: bool = True):
        plotter = RegressionPlotter(data=self.data, treatment=self.treatment, outcome=self.outcome, colors=colors)
        regression_results = [self.run_regressions(control=control) for control in controls]
        return plotter.scatter_plot_with_controls(regression_results=regression_results, axis_number=axis_number, show=show)

    def heatmap_plot_with_controls(self, controls: List[Control], colors: Dict[str, str] = None, axis_number: int = 4, bins: int = 20, rescale_fct_heatmap: Callable = np.sqrt, show: bool = True):
        plotter = RegressionPlotter(data=self.data, treatment=self.treatment, outcome=self.outcome, colors=colors)
        regression_results = [self.run_regressions(control=control) for control in controls]
        return plotter.heatmap_plot_with_controls(regression_results=regression_results, axis_number=axis_number, bins=bins, rescale_fct=rescale_fct_heatmap, show=show)