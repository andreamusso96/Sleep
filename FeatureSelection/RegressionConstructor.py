from enum import Enum
from typing import Union

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objs as go
import numpy as np

from DataPreprocessing.AdminData.AdminDataComplete import AdminData
from DataPreprocessing.GeoData.GeoDataType import GeoDataType
from DataPreprocessing.GeoData.GeoDataComplete import GeoData
from ExpectedBedTime.ExpectedBedTimeAPI import ExpectedBedTime
from typing import List


class FeatureName(Enum):
    DENSITY = 'density'


class FeatureConstructor:
    def __init__(self, admin_data: AdminData, geo_data: GeoData):
        self.admin_data = admin_data
        self.geo_data = geo_data

    def iris_density(self):
        population_data = self.admin_data.get_admin_data()['P19_POP'].to_frame()
        iris_area = self.geo_data.get_areas(geo_data_type=GeoDataType.IRIS).to_frame(name='AREA')
        population_and_area = pd.merge(population_data, iris_area, left_index=True, right_index=True, how='inner')
        density = np.divide(population_and_area['P19_POP'], population_and_area['AREA'], out=np.zeros_like(population_and_area['P19_POP']), where=population_and_area['AREA'] != 0).to_frame(name=FeatureName.DENSITY.value)
        return density

    def city_density(self):
        iris_city_area_population = self._iris_city_area_population()
        city_area_population = iris_city_area_population.groupby(GeoDataType.CITY.value).sum()
        city_density = np.divide(city_area_population['P19_POP'], city_area_population['AREA'],
                                 out=np.zeros_like(city_area_population['P19_POP']),
                                 where=city_area_population['AREA'] != 0).to_frame(name=FeatureName.DENSITY.value)
        return city_density

    def city_density_by_iris(self):
        iris_city_area_population = self._iris_city_area_population()
        city_area_population = iris_city_area_population.groupby(GeoDataType.CITY.value).sum()
        city_density = np.divide(city_area_population['P19_POP'], city_area_population['AREA'],
                                 out=np.zeros_like(city_area_population['P19_POP']),
                                 where=city_area_population['AREA'] != 0).to_frame(name=FeatureName.DENSITY.value)
        city_density_map = {city: density for city, density in zip(city_density.index, city_density[FeatureName.DENSITY.value])}
        iris_city_area_population[FeatureName.DENSITY.value] = iris_city_area_population[GeoDataType.CITY.value].map(city_density_map)
        city_density_by_iris = iris_city_area_population[FeatureName.DENSITY.value].to_frame()
        return city_density_by_iris

    def _iris_city_area_population(self):
        population_data = self.admin_data.get_admin_data()['P19_POP'].to_frame()
        iris_city = self.geo_data.get_matched_geo_data(geometry=GeoDataType.IRIS, other=GeoDataType.CITY).set_index(GeoDataType.IRIS.value)
        iris_area = self.geo_data.get_areas(geo_data_type=GeoDataType.IRIS).to_frame(name='AREA')
        iris_city_area = pd.merge(iris_city, iris_area, left_index=True, right_index=True, how='inner')
        iris_city_area_population = pd.merge(population_data, iris_city_area, left_index=True, right_index=True, how='inner')[[GeoDataType.CITY.value, 'P19_POP', 'AREA']]
        return iris_city_area_population


class Regression:
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.labels = labels
        self.scaler = StandardScaler()
        self.features_scaled = pd.DataFrame(self.scaler.fit_transform(self.features), columns=self.features.columns, index=self.features.index)
        self.label_vals = self.labels.values.flatten()
        self.reg = self._run_regression(features=self.features_scaled, labels=self.label_vals)
        print(self.reg.coef_)

    def plot(self, x_axis: str, color: Union[str, pd.DataFrame] = None, ytickvals: np.ndarray = None, yticktext: List[str] = None):
        fig = go.Figure()
        sorted_features, sorted_labels, sorted_color = self._sort_features(sort_by=x_axis, color=color)
        fig.add_trace(go.Scatter(x=sorted_features[x_axis].values, y=sorted_labels, mode='markers', name='Data', marker=dict(color=sorted_color), text=sorted_features.index))
        fig.add_trace(go.Scatter(x=sorted_features[x_axis].values, y=self.reg.predict(sorted_features), mode='lines', name='Regression'))
        self._layout(fig=fig, x_axis_label=x_axis, ytickvals=ytickvals, yticktext=yticktext)
        fig.show(renderer='browser')

    def _layout(self, fig: go.Figure,  x_axis_label: str, ytickvals: np.ndarray, yticktext: List[str]):
        fig.update_layout(title='Regression', xaxis_title=x_axis_label, yaxis_title='Expected bed time')
        if ytickvals is not None and yticktext is not None:
            fig.update_yaxes(tickvals=ytickvals, ticktext=yticktext)

    def _sort_features(self, sort_by: str, color: Union[str, pd.DataFrame, None]):
        sort_index = np.argsort(self.features_scaled[sort_by].values)
        sorted_features = self.features_scaled.iloc[sort_index]
        sorted_labels = self.label_vals[sort_index]
        sorted_color = self._get_color(color=color, sorted_features=sorted_features, sort_index=sort_index)
        return sorted_features, sorted_labels, sorted_color

    def _get_color(self, color: Union[str, pd.DataFrame, None], sorted_features, sort_index):
        if isinstance(color, pd.DataFrame):
            merged = pd.merge(self.features, color, left_index=True, right_index=True, how='left', suffixes=('_feature', ''))
            merged = merged.iloc[sort_index]
            return merged[color.columns[0]].values
        elif isinstance(color, str):
            return sorted_features[color].values
        else:
            return 'blue'

    @staticmethod
    def _run_regression(features: pd.DataFrame, labels: np.ndarray):
        reg = LinearRegression()
        reg.fit(X=features, y=labels)
        return reg