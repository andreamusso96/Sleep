from datetime import time, date, timedelta
from typing import List, Tuple, Union

import numpy as np
import xarray as xr
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

    def expectation_by_location(self, subset_location: List[str] = None, timespan: Tuple[Union[date, None], Union[date, None]] = None):
        expectation_by_location = self._get_expectation_over_single_dimension(subset_location=subset_location, timespan=timespan, aggregate_over=TrafficDataDimensions.DAY.value)
        expectation_by_location.sort_values(by='session_expectation', inplace=True)
        expectation_by_location = Feature(data=expectation_by_location, name='session_expectation')
        return expectation_by_location

    def expectation_by_day(self, subset_location: List[str] = None, timespan: Tuple[Union[date, None], Union[date, None]] = None):
        expectation_by_day = self._get_expectation_over_single_dimension(subset_location=subset_location, timespan=timespan, aggregate_over=GeoDataType.IRIS.value)
        expectation_by_day.sort_index(inplace=True)
        return expectation_by_day

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

    def time_change_discontinuity_plot(self, z_confidence_interval: float = 1.96, subset_location: List[str] = None):
        time_change = date(2019, 3, 31)
        session_expectation_before_time_change = self.expectation_by_day(subset_location=subset_location, timespan=(None, time_change -timedelta(days=1)))
        session_expectation_after_time_change = self.expectation_by_day(subset_location=subset_location, timespan=(time_change + timedelta(days=1), None))
        confidence_interval_before_time_change = z_confidence_interval * session_expectation_before_time_change['std'].values / np.sqrt(session_expectation_before_time_change['n_obs'].values)
        confidence_interval_after_time_change = z_confidence_interval * session_expectation_after_time_change['std'].values / np.sqrt(session_expectation_after_time_change['n_obs'].values)

        fig = go.Figure()
        x_before = session_expectation_before_time_change.index
        x_after = session_expectation_after_time_change.index
        y_before = session_expectation_before_time_change['session_expectation'].values
        y_after = session_expectation_after_time_change['session_expectation'].values
        fig.add_trace(go.Scatter(x=x_before, y=y_before, error_y=dict(type='data', array=confidence_interval_before_time_change, visible=True), name='Before time change'))
        fig.add_trace(go.Scatter(x=x_after, y=y_after, error_y=dict(type='data', array=confidence_interval_after_time_change, visible=True), name='After time change'))
        fig.add_trace(go.Scatter(x=x_before, y=np.mean(y_before) * np.ones(len(x_before)), name='Before time change average'))
        fig.add_trace(go.Scatter(x=x_after, y=np.mean(y_after) * np.ones(len(x_after)), name='After time change average'))
        fig.update_layout(title='Session expectation before and after time change', template='plotly', xaxis_title='Day of week', yaxis_title='Session expectation')
        fig.show(renderer='browser')

    def _get_expectation_over_single_dimension(self, subset_location: List[str], timespan: Tuple[Union[date, None], Union[date, None]], aggregate_over: str):
        expectation_by_location_and_day = self._get_expectation_by_location_and_day(subset_location=subset_location, timespan=timespan)
        expectation = expectation_by_location_and_day.mean(dim=aggregate_over).T.to_pandas().to_frame(name='session_expectation')
        std = expectation_by_location_and_day.std(dim=aggregate_over).T.to_pandas().to_frame(name='std')
        n_obs = len(expectation_by_location_and_day.coords[aggregate_over].values)
        expectation_std_nobs = pd.concat([expectation, std], axis=1)
        expectation_std_nobs['n_obs'] = n_obs
        return expectation_std_nobs

    def _get_expectation_by_location_and_day(self, subset_location: List[str], timespan: Tuple[Union[date, None], Union[date, None]]):
        data = self._get_selected_data(subset_location=subset_location, timespan=timespan)

        time_since_start = xr.DataArray(np.arange(1, len(data.coords['time'].values) + 1),
                                        dims=TrafficDataDimensions.TIME.value,
                                        coords={TrafficDataDimensions.TIME.value: data.coords['time'].values})
        daily_expectation_by_location = data.dot(time_since_start, dims=TrafficDataDimensions.TIME.value)
        return daily_expectation_by_location

    def _get_selected_data(self, subset_location: List[str], timespan: Tuple[Union[date, None], Union[date, None]]):
        data = self.data.sel(iris=subset_location) if subset_location is not None else self.data
        data = data.sel(day=slice(timespan[0], timespan[1])) if timespan is not None else data
        return data

    def join(self, other):
        assert np.array_equal(self.time_index, other.time_index), 'SessionDistribution can only be joined if they have the same time index'
        return SessionDistribution(session_distribution_data=xr.concat([self.data, other.data], dim=GeoDataType.IRIS.value), start=self.start, end=self.end)