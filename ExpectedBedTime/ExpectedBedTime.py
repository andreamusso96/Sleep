import numpy as np
import pandas as pd
import plotly.graph_objs as go


class ExpectedBedTime:
    def __init__(self, expected_bed_times: pd.DataFrame, time_index: np.ndarray):
        self.expected_bed_times = expected_bed_times
        self.time_index = time_index

    def plot(self, z_confidence: float = 1.96, fig: go.Figure = None, trace_name: str = '', show_plot=True, reference_point: int = 0, show_confidence_intervals=True):
        if fig is None:
            fig = go.Figure()
            fig.update_yaxes(tickvals=np.arange(1, len(self.time_index)), ticktext=self.time_index)
            fig.update_layout(title='Expected bed times', xaxis_title='Iris', yaxis_title='Expected bed time',
                              xaxis_rangeslider_visible=True, font=dict(size=18))

        sorted_expected_bed_times = self.expected_bed_times.sort_values(by='mean_float')
        confidence_intervals = z_confidence * sorted_expected_bed_times['std_float'] / np.sqrt(sorted_expected_bed_times['n_obs'])
        fig.add_trace(go.Scatter(x=sorted_expected_bed_times.index, y=reference_point + sorted_expected_bed_times['mean_float'], error_y=dict(type='data', array=confidence_intervals, visible=show_confidence_intervals), name=f'mean {trace_name}', mode='markers'))
        fig.add_trace(go.Scatter(x=sorted_expected_bed_times.index, y=reference_point + sorted_expected_bed_times['median_float'], name=f'median {trace_name}', mode='markers'))
        if show_plot:
            fig.show(renderer='browser')

    def assign_iris_to_quantile(self, n_quantiles: int) -> pd.DataFrame:
        quantiles = self.expected_bed_times['mean_float'].quantile(np.linspace(0, 1, n_quantiles))
        quantiles[0] = quantiles[0] - 0.0001
        iris_divided_by_quantiles = pd.cut(self.expected_bed_times['mean_float'], bins=quantiles, labels=np.arange(1, n_quantiles)).to_frame(name='quantile')
        iris_divided_by_quantiles.sort_values(by='quantile', inplace=True)
        return iris_divided_by_quantiles