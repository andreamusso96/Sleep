from datetime import time

import numpy as np
import xarray as xr
from typing import List
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd

from DataInterface.GeoDataInterface import GeoData, GeoDataType
from DataInterface.TrafficDataInterface import TrafficDataDimensions
from FeatureExtraction.Feature import Feature


class SessionDistribution:
    def __init__(self, session_distribution_data: xr.DataArray, start: time, end: time):
        super().__init__()
        self.data = session_distribution_data
        self.start = start
        self.end = end
        self.time_index = self.data.coords[TrafficDataDimensions.TIME.value].values

    def expectation_by_location(self, subset: List[str] = None):
        data = self.data.sel(iris=subset) if subset is not None else self.data
        time_since_start = xr.DataArray(np.arange(1, len(data.coords['time'].values) + 1),
                                        dims=TrafficDataDimensions.TIME.value,
                                        coords={TrafficDataDimensions.TIME.value: data.coords['time'].values})
        daily_expectation_by_location = data.dot(time_since_start, dims=TrafficDataDimensions.TIME.value)
        expectation_by_location = daily_expectation_by_location.mean(dim=TrafficDataDimensions.DAY.value).T.to_pandas().to_frame(name='expectation')
        std_by_location = daily_expectation_by_location.std(dim=TrafficDataDimensions.DAY.value).T.to_pandas().to_frame(name='std')
        expectation_by_location = pd.concat([expectation_by_location, std_by_location], axis=1)
        expectation_by_location['n_obs'] = len(data.coords[TrafficDataDimensions.DAY.value].values)
        expectation_by_location.sort_values(by='expectation', inplace=True)
        expectation_by_location = Feature(data=expectation_by_location.rename(columns={'expectation':'session_expectation'}), name='session_expectation')
        return expectation_by_location

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

    def join(self, other):
        assert isinstance(other, SessionDistribution), 'SessionDistribution can only be joined with another SessionDistribution'
        assert np.array_equal(self.time_index, other.time_index), 'SessionDistribution can only be joined if they have the same time index'
        return SessionDistribution(session_distribution_data=xr.concat([self.data, other.data], dim=GeoDataType.IRIS.value), start=self.start, end=self.end)