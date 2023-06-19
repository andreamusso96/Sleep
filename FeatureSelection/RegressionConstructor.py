from enum import Enum
from typing import Union

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from DataInterface.AdminDataInterface import AdminData
from DataInterface.GeoDataInterface import GeoData, GeoDataType
from ExpectedBedTime.ExpectedBedTimeAPI import ExpectedBedTime
from typing import List


class FeatureConstructor:
    def __init__(self, admin_data: AdminData, geo_data: GeoData):
        self.admin_data = admin_data
        self.geo_data = geo_data

    def iris_centrality(self, iris: List[str]):
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=iris, other_geo_data_types=GeoDataType.CITY).to_crs(epsg=2154)
        city_geo_data = iris_geo_data[[GeoDataType.CITY.value, 'geometry']].dissolve(by=GeoDataType.CITY.value).reset_index()
        city_geo_data['centroid'] = city_geo_data['geometry'].centroid
        iris_geo_data['centroid'] = iris_geo_data['geometry'].centroid
        iris_city_centroids = iris_geo_data.merge(city_geo_data, left_on=GeoDataType.CITY.value, right_on=GeoDataType.CITY.value, how='left', suffixes=('_iris', '_city'))[[GeoDataType.IRIS.value, 'centroid_iris', 'centroid_city']]
        iris_city_centroids['distance'] = iris_city_centroids.apply(lambda row: row['centroid_iris'].distance(row['centroid_city']), axis=1)
        iris_distance_to_city_center = iris_city_centroids[[GeoDataType.IRIS.value, 'distance']].set_index(GeoDataType.IRIS.value)
        return iris_distance_to_city_center.rename(columns={'distance': 'centrality'})

    def iris_density_of_city(self, iris: List[str]):
        population_data = self.admin_data.get_admin_data(subset=iris)['P19_POP'].to_frame()
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=iris, other_geo_data_types=GeoDataType.CITY).to_crs(epsg=2154).set_index(GeoDataType.IRIS.value)
        iris_geo_data['AREA'] = iris_geo_data.area
        iris_city_area_population = pd.merge(population_data, iris_geo_data, left_index=True, right_index=True, how='inner')[[GeoDataType.CITY.value, 'P19_POP', 'AREA']]
        city_area_population = iris_city_area_population.groupby(GeoDataType.CITY.value).sum()
        city_density = np.divide(city_area_population['P19_POP'], city_area_population['AREA'],
                                 out=np.zeros_like(city_area_population['P19_POP']),
                                 where=city_area_population['AREA'] != 0).to_frame(name='density')
        city_density_map = {city: density for city, density in
                            zip(city_density.index, city_density['density'])}
        iris_city_area_population['density'] = iris_city_area_population[GeoDataType.CITY.value].map(city_density_map)
        iris_density_of_city = iris_city_area_population[['density']]
        return iris_density_of_city

    def iris_business_density(self, iris: List[str]):
        codes_businesses_likely_to_be_open_at_night = ['A101', 'A104', 'A504', 'B101', 'B201', 'B204', 'B316', 'C701', 'D101', 'D102', 'D103', 'D104', 'D105', 'D106', 'D107', 'D111', 'D303', 'E102', 'E107', 'F303', 'G102']
        business_data = self.admin_data.get_admin_data(coarsened_equip=False, subset=iris)[[f'EQUIP_{code}' for code in codes_businesses_likely_to_be_open_at_night]]
        business_data['business_count'] = business_data.sum(axis=1)
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=iris).to_crs(epsg=2154)
        iris_geo_data['AREA'] = iris_geo_data.area
        business_counts_and_area = pd.merge(business_data['business_count'].to_frame(), iris_geo_data.set_index(GeoDataType.IRIS.value), left_index=True, right_index=True, how='inner')
        density = np.divide(business_counts_and_area['business_count'], business_counts_and_area['AREA'], out=np.zeros_like(business_counts_and_area['business_count']), where=business_counts_and_area['AREA'] != 0).to_frame(name='business_density')
        return density

    def iris_shares(self, iris: List[str], var_names: Union[str, List[str]]):
        if isinstance(var_names, str):
            var_names = [var_names]
        admin_data_iris = self.admin_data.get_admin_data(subset=iris, coarsened_equip=False, selected_pop_vars=False)
        population = admin_data_iris['P19_POP'].to_frame()
        var_vals = admin_data_iris[var_names]
        var_share = np.divide(var_vals.values, population.values, out=np.zeros_like(var_vals), where=population.values != 0)
        var_share = pd.DataFrame(var_share, columns=var_vals.columns, index=var_vals.index)
        return var_share

    def iris_density(self, iris: List[str], var_names: Union[str, List[str]]):
        iris_area = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=iris).to_crs(epsg=2154).set_index(GeoDataType.IRIS.value).area.to_frame(name='AREA')
        var_vals = self.admin_data.get_admin_data(subset=iris, coarsened_equip=False, selected_pop_vars=False)[var_names]
        var_vals_and_area = pd.merge(var_vals, iris_area, left_index=True, right_index=True, how='inner')

        var_vals = var_vals_and_area[var_names]
        area = var_vals_and_area['AREA'].to_frame()
        var_densities = np.divide(var_vals.values, area.values, out=np.zeros_like(var_vals), where=area.values != 0)
        var_densities = pd.DataFrame(var_densities, columns=var_vals.columns, index=var_vals.index)
        return var_densities




class Regression:
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.labels = labels
        self.scaler = StandardScaler()
        self.features_scaled = pd.DataFrame(self.scaler.fit_transform(self.features), columns=self.features.columns, index=self.features.index)
        self.label_vals = self.labels.values.flatten()
        self.reg = self._run_regression(features=self.features_scaled, labels=self.label_vals)

    def plot(self, x_axis: str, title: str = 'Regression', color: Union[str, pd.DataFrame] = None, ytickvals: np.ndarray = None, yticktext: List[str] = None):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Correlation', 'Regression Table'), specs=[[{"type": "scatter"}, {"type": "table"}]], column_widths=[0.7, 0.3])
        trace_data, trace_regression_forecasts = self._plot_correlation_plot(x_axis=x_axis, color=color)
        trace_table = self._plot_table()

        fig.add_trace(trace_data, row=1, col=1)
        fig.add_trace(trace_regression_forecasts, row=1, col=1)
        fig.add_trace(trace_table, row=1, col=2)
        self._layout(fig=fig, x_axis_label=x_axis, ytickvals=ytickvals, yticktext=yticktext, title=title)
        fig.show(renderer='browser')

    def _plot_table(self):
        headers, rows = self._prepare_table()
        column_widths = np.ones(len(headers))
        column_widths[0] = 2
        trace_table = go.Table(header=dict(values=headers), cells=dict(values=[x for x in zip(*rows)]), columnwidth=column_widths)
        return trace_table

    def _plot_correlation_plot(self, x_axis: str, color: Union[str, pd.DataFrame]):
        sorted_features, sorted_labels, sorted_color = self._sort_features(sort_by=x_axis, color=color)
        x_vals = sorted_features[x_axis].values
        trace_data = go.Scatter(x=x_vals, y=sorted_labels, mode='markers', name='Data', marker=dict(color=sorted_color), text=sorted_features.index)
        trace_regression_forecasts = go.Scatter(x=x_vals, y=self.reg.params[x_axis] * x_vals + self.reg.params['const'], mode='markers', marker=dict(symbol='x'), name='Regression')
        return trace_data, trace_regression_forecasts

    def _layout(self, fig: go.Figure,  x_axis_label: str, ytickvals: np.ndarray, yticktext: List[str], title: str):
        fig.update_layout(title=title, xaxis_title=x_axis_label, yaxis_title='Expected bed time', template='plotly')
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

    def _prepare_table(self):
        headers = ['Variable', 'Coefficient', 'P-value', 'T-value']
        rows = []
        for i in range(len(self.reg.params)):
            var_name = self.reg.params.index[i]
            rows.append([f"{var_name}", Regression._format_float(x=self.reg.params[i]), Regression._format_float(x=self.reg.pvalues[i]), Regression._format_float(x=self.reg.tvalues[i])])

        return headers, rows

    @staticmethod
    def _format_float(x):
        return '%s' % float('%.3g' % x)

    @staticmethod
    def _run_regression(features: pd.DataFrame, labels: np.ndarray):
        features = add_constant(features)
        ols = OLS(endog=labels, exog=features, hasconst=True)
        results = ols.fit()
        return results