from typing import Union, List, Dict
from itertools import accumulate
import operator

from sklearn.preprocessing import StandardScaler
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
    def __init__(self, regression: RegressionResultsSM, treatment: str, controls: List[Control]):
        self.effect = regression.params[treatment]
        self.const = regression.params['const']
        self.conf_high = regression.conf_int(alpha=0.05).loc[treatment].values[1]
        self.conf_low = regression.conf_int(alpha=0.05).loc[treatment].values[0]
        self.p_value = regression.pvalues[treatment]
        self._regression = regression
        self.treatment = treatment
        self.controls = controls

    def to_dataframe(self):
        return pd.DataFrame({'effect': self.effect, 'conf_high': self.conf_high, 'conf_low': self.conf_low, 'pvalue': self.p_value, 'treatment': self.treatment, 'controls': " + ".join([c.name for c in self.controls])}, index=[0])


class Regression:
    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str):
        self.data = data.sort_values(by=treatment)
        self.treatment = treatment
        self.outcome = outcome

    def run_regressions_with_cumulative_controls(self, controls: List[Control]):
        effects = []
        for i in range(1, len(controls) + 1):
            temp_controls = controls[:i]
            reg_result = self.run_regression(controls=temp_controls)
            effects.append(reg_result.to_dataframe())

        effects = pd.concat(effects, axis=0)
        return effects

    def run_regression(self, controls: List[Control]):
        independent_vars = [self.treatment]
        for control in controls:
            independent_vars += control.var_names
        regression = self._ols_regression(independent_vars=independent_vars, dependent_var=self.outcome)
        result = RegressionResult(regression=regression, treatment=self.treatment, controls=controls)
        return result

    def _ols_regression(self, independent_vars, dependent_var):
        features = add_constant(self.data[independent_vars])
        ols = OLS(endog=self.data[dependent_var], exog=features, hasconst=True)
        results = ols.fit()
        return results

    def scatter_plot_with_controls(self, controls: List[Control], cumulative: bool = False, axis_number: int = 4, colors: Dict[str, str] = None, show: bool = True):
        if cumulative:
            effects = self.run_regressions_with_cumulative_controls(controls=controls)
        else:
            effects = self.run_regression(controls=controls).to_dataframe()

        trace_control_bar_chart = self._control_bar_chart_trace(effects=effects, controls=controls, axis_number=axis_number, colors=colors, show_controls_at_x=show)
        trace_scatter_data, trace_regression_line = self._scatter_plot_trace(x_axis=self.data[self.treatment].values, y_axis=self.data[self.outcome].values, regression_result=self.run_regression(controls=[]))
        if show:
            trace_table = self._table_trace(effects=effects)
            self._show_scatter_plot_with_controls(trace_scatter_data=trace_scatter_data, trace_regression_line=trace_regression_line,
                                                  trace_bar_chart_controls=trace_control_bar_chart, trace_table=trace_table)
        else:
            return trace_control_bar_chart, trace_scatter_data, trace_regression_line

    def _show_scatter_plot_with_controls(self, trace_scatter_data, trace_regression_line, trace_bar_chart_controls, trace_table):
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.03, specs=[[{'rowspan': 2}, {}], [None, {'type': 'table'}]], subplot_titles=('Scatter plot', f'Effects of {self.treatment} on {self.outcome}',  f'Dependent variable {self.outcome}'), column_widths=[0.6, 0.4])
        fig.add_trace(trace_scatter_data, row=1, col=1)
        fig.add_trace(trace_regression_line, row=1, col=1)
        fig.add_trace(trace_bar_chart_controls, row=1, col=2)
        fig.add_trace(trace_table, row=2, col=2)
        self._setup_layout(fig=fig)
        fig.show(renderer='browser')

    def _setup_layout(self, fig):
        fig.update_xaxes(title_text=self.treatment, showline=True, linecolor='black', linewidth=2, row=1, col=1)
        fig.update_yaxes(title_text=self.outcome, showline=True, linecolor='black', linewidth=2, row=1, col=1)
        fig.update_xaxes(title_text='Control', showline=True, linecolor='black', linewidth=2, row=1, col=2)
        fig.update_yaxes(title_text='Effect', showline=True, linecolor='black', linewidth=2, row=1, col=2)
        fig.update_layout(template='plotly_white', showlegend=False)

    @staticmethod
    def _scatter_plot_trace(x_axis: np.ndarray, y_axis: np.ndarray, regression_result: RegressionResult):
        trace_scatter_data = go.Scatter(x=x_axis, y=y_axis, mode='markers', marker=dict(color='grey', opacity=0.3), showlegend=False)
        trace_regression_line = go.Scatter(x=x_axis, y=x_axis * regression_result.effect + regression_result.const, mode='lines', line=dict(color='black', width=2), showlegend=False)
        return trace_scatter_data, trace_regression_line

    @staticmethod
    def _control_bar_chart_trace(effects: pd.DataFrame, controls: List[Control], axis_number: int, colors: Dict[str, str], show_controls_at_x: bool):
        ys = effects['effect']
        xs = effects['controls'] if show_controls_at_x else list(accumulate([' ' for _ in range(len(controls))], operator.add))
        error_y = dict(type='data', symmetric=False, array=effects['conf_high'] - effects['effect'], arrayminus=effects['effect'] - effects['conf_low'])
        colors_ = [colors[c.name] for c in controls] if colors is not None else [c for c in px.colors.qualitative.Plotly][:len(controls)]
        trace_bar_chart = go.Bar(x=xs, y=ys, error_y=error_y, xaxis=f'x{axis_number}', yaxis=f'y{axis_number}', marker=dict(color=colors_), showlegend=False)
        return trace_bar_chart

    @staticmethod
    def _table_trace(effects):
        table = effects[['controls', 'effect', 'pvalue']] #.applymap(columns=['effect', 'pvalue'], func=Regression._format_float)
        column_widths = np.ones(len(table.columns))
        column_widths[0] = 2
        trace_table = go.Table(header=dict(values=table.columns), cells=dict(values=table.values.T), columnwidth=column_widths)
        return trace_table

    @staticmethod
    def _format_float(x):
        return '%s' % float('%.3g' % x)