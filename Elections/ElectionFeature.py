from enum import Enum
import os
import webbrowser
from typing import List

import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go

from DataInterface.GeoDataInterface import GeoData, GeoDataType


class ElectionFeatureName(Enum):
    ENTROPY = 'entropy'
    SIMPSON = 'simpson'


class ElectionFeature:
    def __init__(self, election_result_by_location: pd.DataFrame):
        self.result = election_result_by_location

    def geo_plot(self, geo_data: GeoData, feature: ElectionFeatureName, subset: List[str] = None):
        cmap = 'plasma'
        feature_by_location = self.get_election_feature(feature=feature, subset=subset)
        iris_geo_data = geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset).set_index(GeoDataType.IRIS.value)
        feature_with_geo_data = gpd.GeoDataFrame(feature_by_location.merge(iris_geo_data, left_index=True, right_index=True))[['geometry', feature.value]]
        html_map = feature_with_geo_data.explore(column=feature.value, legend=True, tiles="CartoDB positron", cmap=cmap)
        html_map.save(f'temp/election_result_{feature.value}_map.html')
        webbrowser.open('file://' + os.path.realpath(f'temp/election_result_{feature.value}_map.html'))

    def feature_by_location_scatter_plot(self, feature: ElectionFeatureName, subset: List[str] = None):
        feature_by_location = self.get_election_feature(feature=feature, subset=subset)
        feature_by_location = feature_by_location.sort_values(by=feature.value)
        fig = go.Figure()
        trace_feature_by_location_scatter = go.Scatter(x=feature_by_location.index, y=feature_by_location[feature.value], mode='markers', name=feature.value)
        fig.add_trace(trace_feature_by_location_scatter)
        fig.update_layout(title=f'{feature.name} by location', xaxis_title='Iris', yaxis_title=feature.name,
                          xaxis_rangeslider_visible=True, font=dict(size=18), template='plotly')
        fig.show(renderer='browser')

    def get_election_feature(self, feature: ElectionFeatureName, subset: List[str] = None):
        result = self.result.loc[subset] if subset is not None else self.result
        if feature == ElectionFeatureName.ENTROPY:
            return self._get_entropy(result=result)
        elif feature == ElectionFeatureName.SIMPSON:
            return self._get_simpson(result=result)
        else:
            raise NotImplementedError

    @staticmethod
    def _get_entropy(result: pd.DataFrame):
        entropy = result.apply(ElectionFeature.entropy, axis=1).to_frame(name=ElectionFeatureName.ENTROPY.value)
        return entropy

    @staticmethod
    def _get_simpson(result: pd.DataFrame):
        simpson = result.apply(ElectionFeature.simpson, axis=1).to_frame(name=ElectionFeatureName.SIMPSON.value)
        return simpson

    @staticmethod
    def entropy(x):
        vals = x.values
        return -1 * np.dot(vals, np.log(vals, out=np.zeros_like(vals), where=vals != 0))

    @staticmethod
    def simpson(x):
        vals = x.values
        return 1 - np.dot(vals, vals)
