import webbrowser
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import geopandas as gpd

from DataPreprocessing.GeoData.GeoData import IrisGeoData


class ExpectedBedTime:
    def __init__(self, data: pd.DataFrame, time_index: np.ndarray):
        self.data = data
        self.time_index = time_index

    def plot(self, z_confidence: float = 1.96, fig: go.Figure = None, trace_name: str = '', show_plot=True, reference_point: int = 0, show_confidence_intervals=True):
        if fig is None:
            fig = go.Figure()
            fig.update_yaxes(tickvals=np.arange(1, len(self.time_index)), ticktext=self.time_index)
            fig.update_layout(title='Expected bed times', xaxis_title='Iris', yaxis_title='Expected bed time',
                              xaxis_rangeslider_visible=True, font=dict(size=18))

        sorted_expected_bed_times = self.data.sort_values(by='mean_float')
        confidence_intervals = z_confidence * sorted_expected_bed_times['std_float'] / np.sqrt(sorted_expected_bed_times['n_obs'])
        fig.add_trace(go.Scatter(x=sorted_expected_bed_times.index, y=reference_point + sorted_expected_bed_times['mean_float'], error_y=dict(type='data', array=confidence_intervals, visible=show_confidence_intervals), name=f'mean {trace_name}', mode='markers'))
        fig.add_trace(go.Scatter(x=sorted_expected_bed_times.index, y=reference_point + sorted_expected_bed_times['median_float'], name=f'median {trace_name}', mode='markers'))
        if show_plot:
            fig.show(renderer='browser')

    def geo_plot(self, iris_geo_data: IrisGeoData, n_quantiles: int = None):
        if n_quantiles is not None:
            bed_time_data = self.assign_iris_to_quantile(n_quantiles=n_quantiles)
            column = 'quantile'
            cmap = 'tab10'
        else:
            bed_time_data = self.data
            column = 'mean_float'
            cmap = 'plasma'

        bed_times_with_geo_data = gpd.GeoDataFrame(bed_time_data.merge(iris_geo_data.data.set_index('iris'), left_index=True, right_index=True, how='left'))[[column, 'geometry']]
        html_map = bed_times_with_geo_data.explore(column=column, legend=True, tiles="CartoDB positron", cmap=cmap)
        html_map.save('temp/expected_bed_time_map.html')
        webbrowser.open('file://' + os.path.realpath('temp/expected_bed_time_map.html'))

    def assign_iris_to_quantile(self, n_quantiles: int) -> pd.DataFrame:
        quantiles = self.data['mean_float'].quantile(np.linspace(0, 1, n_quantiles))
        quantiles[0] = quantiles[0] - 0.0001
        iris_divided_by_quantiles = pd.cut(self.data['mean_float'], bins=quantiles, labels=np.arange(1, n_quantiles)).to_frame(name='quantile')
        iris_divided_by_quantiles.sort_values(by='quantile', inplace=True)
        return iris_divided_by_quantiles

    def join(self, other):
        assert isinstance(other, ExpectedBedTime), 'ExpectedBedTime can only be joined with another ExpectedBedTime'
        assert np.array_equal(self.time_index, other.time_index), 'ExpectedBedTime can only be joined if they have the same time index'
        return ExpectedBedTime(data=pd.concat([self.data, other.data], axis=0), time_index=self.time_index)