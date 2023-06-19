from typing import List, Union

import pandas as pd
import numpy as np

from FeatureExtraction.FeatureCalculator import FeatureCalculator
from FeatureExtraction.Feature import Feature
from DataInterface.AdminDataInterface import AdminData
from DataInterface.GeoDataInterface import GeoData, GeoDataType


class IrisFeatureCalculator(FeatureCalculator):
    def __init__(self, admin_data: AdminData, geo_data: GeoData):
        super().__init__()
        self.admin_data = admin_data
        self.geo_data = geo_data

    def centrality(self, subset: List[str]) -> Feature:
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset, other_geo_data_types=GeoDataType.CITY).to_crs(epsg=2154)
        city_geo_data = iris_geo_data[[GeoDataType.CITY.value, 'geometry']].dissolve(by=GeoDataType.CITY.value).reset_index()
        city_geo_data['centroid'] = city_geo_data['geometry'].centroid
        iris_geo_data['centroid'] = iris_geo_data['geometry'].centroid
        iris_city_centroids = iris_geo_data.merge(city_geo_data, left_on=GeoDataType.CITY.value, right_on=GeoDataType.CITY.value, how='left', suffixes=('_iris', '_city'))[[GeoDataType.IRIS.value, 'centroid_iris', 'centroid_city']]
        iris_city_centroids['distance'] = iris_city_centroids.apply(lambda row: row['centroid_iris'].distance(row['centroid_city']), axis=1)
        iris_distance_to_city_center = iris_city_centroids[[GeoDataType.IRIS.value, 'distance']].set_index(GeoDataType.IRIS.value)
        iris_distance_to_city_center.rename(columns={'distance': 'centrality'}, inplace=True)
        iris_distance_to_city_center = Feature(data=iris_distance_to_city_center, name='centrality')
        return iris_distance_to_city_center

    def density_of_city(self, subset: List[str]) -> Feature:
        population_data = self.admin_data.get_admin_data(subset=subset)['P19_POP'].to_frame()
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset, other_geo_data_types=GeoDataType.CITY).to_crs(epsg=2154).set_index(GeoDataType.IRIS.value)
        iris_geo_data['AREA'] = iris_geo_data.area
        iris_city_area_population = pd.merge(population_data, iris_geo_data, left_index=True, right_index=True, how='inner')[[GeoDataType.CITY.value, 'P19_POP', 'AREA']]
        city_area_population = iris_city_area_population.groupby(GeoDataType.CITY.value).sum()
        city_density = np.divide(city_area_population['P19_POP'], city_area_population['AREA'], out=np.zeros_like(city_area_population['P19_POP']), where=city_area_population['AREA'] != 0).to_frame(name='density')
        city_density_map = {city: density for city, density in zip(city_density.index, city_density['density'])}
        iris_city_area_population['density'] = iris_city_area_population[GeoDataType.CITY.value].map(city_density_map)
        iris_density_of_city = Feature(data=iris_city_area_population[['density']].rename(columns={'density': 'density_of_city'}), name='density_of_city')
        return iris_density_of_city

    def business_density(self, subset: List[str]) -> Feature:
        codes_businesses_likely_to_be_open_at_night = ['A101', 'A104', 'A504', 'B101', 'B201', 'B204', 'B316',
                                                       'C701', 'D101', 'D102', 'D103', 'D104', 'D105', 'D106',
                                                       'D107', 'D111', 'D303', 'E102', 'E107', 'F303', 'G102']
        business_data = self.admin_data.get_admin_data(coarsened_equip=False, subset=subset)[[f'EQUIP_{code}' for code in codes_businesses_likely_to_be_open_at_night]]
        business_data['business_count'] = business_data.sum(axis=1)
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset).to_crs(epsg=2154)
        iris_geo_data['AREA'] = iris_geo_data.area
        business_counts_and_area = pd.merge(business_data['business_count'].to_frame(), iris_geo_data.set_index(GeoDataType.IRIS.value), left_index=True, right_index=True, how='inner')
        density = np.divide(business_counts_and_area['business_count'], business_counts_and_area['AREA'], out=np.zeros_like(business_counts_and_area['business_count']), where=business_counts_and_area['AREA'] != 0).to_frame(name='business_density')
        iris_business_density = Feature(data=density, name='business_density')
        return iris_business_density

    def var_shares(self, subset: List[str], var_names: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(var_names, str):
            var_names = [var_names]
        admin_data_iris = self.admin_data.get_admin_data(subset=subset, coarsened_equip=False, selected_pop_vars=False)
        population = admin_data_iris['P19_POP'].to_frame()
        var_vals = admin_data_iris[var_names]
        var_share = np.divide(var_vals.values, population.values, out=np.zeros_like(var_vals), where=population.values != 0)
        var_share = pd.DataFrame(var_share, columns=var_vals.columns, index=var_vals.index)
        return var_share

    def var_density(self, subset: List[str], var_names: Union[str, List[str]]) -> pd.DataFrame:
        iris_area = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset).to_crs(epsg=2154).set_index(GeoDataType.IRIS.value).area.to_frame(name='AREA')
        var_vals = self.admin_data.get_admin_data(subset=subset, coarsened_equip=False, selected_pop_vars=False)[var_names]
        var_vals_and_area = pd.merge(var_vals, iris_area, left_index=True, right_index=True, how='inner')
        var_vals = var_vals_and_area[var_names]
        area = var_vals_and_area['AREA'].to_frame()
        var_densities = np.divide(var_vals.values, area.values, out=np.zeros_like(var_vals), where=area.values != 0)
        var_densities = pd.DataFrame(var_densities, columns=var_vals.columns, index=var_vals.index)
        return var_densities