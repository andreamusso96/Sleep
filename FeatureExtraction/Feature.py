import os
import webbrowser
from typing import List

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

from DataInterface.GeoDataInterface import GeoData, GeoDataType
from DataInterface.AdminDataInterface import AdminData


class Feature:
    def __init__(self, data: pd.DataFrame, name: str):
        self.name = name
        self.data = data

    def geo_plot(self, geo_data: GeoData, subset: List[str] = None, log: bool = False, normalize: bool = False, remove_outliers: bool = False, show: bool = True):
        feature_with_geo_data = self.get_feature_with_geo_data(geo_data=geo_data, subset=subset)
        feature_with_geo_data = self._format_data_for_geo_plot(feature_with_geo_data=feature_with_geo_data, log=log, normalize=normalize, remove_outliers=remove_outliers)
        html_map = feature_with_geo_data.explore(column=self.name, legend=True, tiles="CartoDB positron", cmap='plasma')
        if show:
            html_map.save(f'temp/{self.name}_map.html')
            webbrowser.open('file://' + os.path.realpath(f'temp/{self.name}_map.html'))
        return html_map

    def get_feature_with_geo_data(self, geo_data: GeoData, subset: List[str] = None):
        iris_geo_data = geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset).set_index(
            GeoDataType.IRIS.value)
        feature_with_geo_data = gpd.GeoDataFrame(self.data.merge(iris_geo_data, left_index=True, right_index=True))[
            ['geometry', self.name]]
        return feature_with_geo_data

    def _format_data_for_geo_plot(self, feature_with_geo_data, log: bool, normalize: bool, remove_outliers: bool):
        if log:
            feature_with_geo_data[self.name] = self._log(data=feature_with_geo_data, col=self.name)
        if normalize:
            scaler = StandardScaler()
            feature_with_geo_data[self.name] = scaler.fit_transform(feature_with_geo_data[self.name].values.reshape(-1, 1)).flatten()
            if remove_outliers:
                mask = np.abs(feature_with_geo_data[self.name]) < 2.5
                feature_with_geo_data = feature_with_geo_data.loc[mask]

        return feature_with_geo_data

    def scatter_plot(self, subset: List[str] = None, confidence_intervals: bool = False, z_confidence: float = 1.96, log: bool = False):
        data = self.data.loc[subset] if subset is not None else self.data
        data = data.sort_values(by=self.name)
        if log:
            data[self.name] = self._log(data=data, col=self.name)

        fig = go.Figure()
        if confidence_intervals:
            assert 'std' in self.data.columns and 'n_obs' in self.data.columns, 'Standard deviation or number of observations not found in data'
            confidence_intervals = z_confidence * data['std'] / np.sqrt(data['n_obs'])
            error_y = dict(type='data', array=confidence_intervals, visible=True)
        else:
            error_y = None

        trace_feature_by_location = go.Scatter(x=data.index, y=data[self.name], mode='markers', name=self.name, error_y=error_y)
        fig.add_trace(trace_feature_by_location)
        fig.update_layout(title=f'{self.name} by location', xaxis_title='Iris', yaxis_title=self.name, xaxis_rangeslider_visible=True, font=dict(size=18), template='plotly')
        fig.show(renderer='browser')

    def correlation_plot(self, covariates: pd.DataFrame, save: bool = True):
        merged_data = pd.merge(self.data[self.name].to_frame(), covariates, left_index=True, right_index=True, how='inner')
        correlations = merged_data.corr()[[self.name]]
        correlations.sort_values(by=self.name, inplace=True)
        correlations.dropna(inplace=True)
        correlations.drop(index=self.name, inplace=True)
        fig = go.Figure()
        trace_correlation = go.Bar(x=correlations.index, y=correlations[self.name], name='Correlation')
        fig.add_trace(trace_correlation)
        fig.update_layout(title=f'Correlation of {self.name} with covariates', xaxis_title='Covariates', yaxis_title='Correlation', font=dict(size=18), template='plotly', xaxis_rangeslider_visible=True)
        fig.show(renderer='browser')
        if save:
            fig.write_html(f'temp/{self.name.lower().replace(" ", "_")}_correlation.html')

    def log_data(self):
        return pd.DataFrame(data=self._log(data=self.data, col=self.name), index=self.data.index, columns=[self.name])

    @staticmethod
    def _log(data: pd.DataFrame, col: str):
        vals = data[col].values
        return np.log(vals, out=np.zeros_like(vals), where=vals > 0)