from datetime import time
import webbrowser
import os

import numpy as np
import xarray as xr
from typing import List
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import geopandas as gpd

from DataPreprocessing.GeoData.GeoDataComplete import GeoData
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
from Utils import TrafficDataDimensions


class SessionDistribution:
    def __init__(self, session_distribution_data: xr.DataArray, start: time, end: time):
        self.data = session_distribution_data
        self.start = start
        self.end = end
        self.time_index = self.data.coords[TrafficDataDimensions.TIME.value].values

    def geo_plot(self, geo_data: GeoData, iris: List[str] = None):
        cmap = 'plasma'
        expectation_by_location = self.compute_expectation_by_location(iris=iris)
        expectations_with_geo_data = gpd.GeoDataFrame(expectation_by_location.merge(geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=iris).set_index(GeoDataType.IRIS.value), left_index=True, right_index=True))[['geometry', 'expectation']]
        html_map = expectations_with_geo_data.explore(column='expectation', legend=True, tiles="CartoDB positron", cmap=cmap)
        html_map.save('temp/session_distribution_map.html')
        webbrowser.open('file://' + os.path.realpath('temp/session_distribution_map.html'))

    def distribution_plot(self, iris: List[str] = None):
        data = self.data.sel(iris=iris) if iris is not None else self.data
        session_distribution_day_average = data.mean(dim=TrafficDataDimensions.DAY.value).T.to_pandas()
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Session Distribution By Location", "Session Distribution Average Across Locations"))
        heatmap_session_distribution_by_location = go.Heatmap(z=session_distribution_day_average.values.T, y=session_distribution_day_average.columns, x=session_distribution_day_average.index, showscale=False)
        fig.add_trace(heatmap_session_distribution_by_location, row=1, col=1)
        all_distribution_trace = [go.Scatter(x=session_distribution_day_average.index, y=session_distribution_day_average[col].values, opacity=0.1, mode='lines', line=dict(color='red'), name=col) for col in session_distribution_day_average.columns]
        distribution_trace = go.Scatter(x=session_distribution_day_average.index, y=session_distribution_day_average.mean(axis=1), mode='lines', line=dict(color='blue', width=5), name='Average')
        fig.add_traces(all_distribution_trace + [distribution_trace], rows=1, cols=2)
        fig.update_layout(title='Session distribution plots', template='plotly')
        fig.update_yaxes(title_text='Location', row=1, col=1)
        fig.update_xaxes(title_text='Time of day', row=1, col=1)
        fig.update_yaxes(title_text='Probability of Session', row=1, col=2)
        fig.update_xaxes(title_text='Time of day', row=1, col=2)
        fig.show(renderer='browser')

    def expectation_by_location_plot(self, z_confidence: float = 1.96, iris: List[str] = None, show_confidence_intervals: bool = True):
        expectation_by_location = self.compute_expectation_by_location(iris=iris)
        confidence_intervals = z_confidence * expectation_by_location['std'] / np.sqrt(expectation_by_location['n_obs'])
        fig = go.Figure()
        trace_expected_session_time = go.Scatter(x=expectation_by_location.index, y=expectation_by_location['expectation'], mode='markers', name='Expected Session Time', error_y=dict(type='data', array=confidence_intervals, visible=show_confidence_intervals))
        fig.add_trace(trace_expected_session_time)
        fig.update_layout(title='Expected session time', xaxis_title='Iris', yaxis_title='Expected session',
                          xaxis_rangeslider_visible=True, font=dict(size=18), template='plotly')
        fig.update_yaxes(title_text='Expected Session Timem', tickvals=np.arange(1, len(self.time_index) + 1), ticktext=self.time_index)
        fig.update_xaxes(title_text='Location')
        fig.show(renderer='browser')

    def compute_expectation_by_location(self, iris: List[str] = None):
        data = self.data.sel(iris=iris) if iris is not None else self.data
        time_since_start = xr.DataArray(np.arange(1, len(data.coords['time'].values) + 1), dims=TrafficDataDimensions.TIME.value, coords={TrafficDataDimensions.TIME.value: data.coords['time'].values})
        daily_expectation_by_location = data.dot(time_since_start, dims=TrafficDataDimensions.TIME.value)
        expectation_by_location = daily_expectation_by_location.mean(dim=TrafficDataDimensions.DAY.value).T.to_pandas().to_frame(name='expectation')
        std_by_location = daily_expectation_by_location.std(dim=TrafficDataDimensions.DAY.value).T.to_pandas().to_frame(name='std')
        expectation_by_location = pd.concat([expectation_by_location, std_by_location], axis=1)
        expectation_by_location['n_obs'] = len(data.coords[TrafficDataDimensions.DAY.value].values)
        expectation_by_location.sort_values(by='expectation', inplace=True)
        return expectation_by_location

    def join(self, other):
        assert isinstance(other, SessionDistribution), 'SessionDistribution can only be joined with another SessionDistribution'
        assert np.array_equal(self.time_index, other.time_index), 'SessionDistribution can only be joined if they have the same time index'
        return SessionDistribution(session_distribution_data=xr.concat([self.data, other.data], dim=GeoDataType.IRIS.value), start=self.start, end=self.end)