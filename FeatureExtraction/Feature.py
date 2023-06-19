import os
import webbrowser
from typing import List

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from DataInterface.GeoDataInterface import GeoData, GeoDataType


class Feature:
    def __init__(self, data: pd.DataFrame, name: str):
        self.name = name
        self.data = data

    def geo_plot(self, geo_data: GeoData, subset: List[str] = None):
        iris_geo_data = geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset).set_index(GeoDataType.IRIS.value)
        feature_with_geo_data = gpd.GeoDataFrame(self.data.merge(iris_geo_data, left_index=True, right_index=True))[['geometry', self.name]]
        html_map = feature_with_geo_data.explore(column=self.name, legend=True, tiles="CartoDB positron", cmap='plasma')
        html_map.save(f'temp/{self.name}_map.html')
        webbrowser.open('file://' + os.path.realpath(f'temp/{self.name}_map.html'))

    def scatter_plot(self, subset: List[str] = None, confidence_intervals: bool = False, z_confidence: float = 1.96):
        data = self.data.loc[subset] if subset is not None else self.data
        data = data.sort_values(by=self.name)
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