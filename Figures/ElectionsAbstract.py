import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from DataInterface.GeoDataInterface import GeoData, GeoDataType
from DataInterface.AdminDataInterface import AdminData
from DataInterface.ElectionDataInterface import ElectionData
from FeatureExtraction.ElectionFeatureCalculator import ElectionFeatureCalculator, ElectionFeatureName, Party
from FeatureExtraction.ServiceConsumptionFeatureCalculator import ServiceConsumptionFeatureCalculator
from FeatureExtraction.ServiceConsumption.ServiceConsumption import ServiceConsumption
from FeatureExtraction.Feature import Feature
from config import FIGURE_PATH


class FigureElectionsAbstract:
    def __init__(self, geo_data: GeoData, admin_data: AdminData, election_data: ElectionData, service_consumption: ServiceConsumption):
        self.election_feature_calculator = ElectionFeatureCalculator(election_data=election_data, geo_data=geo_data)
        self.service_consumption_feature_calculator = ServiceConsumptionFeatureCalculator(service_consumption=service_consumption, admin_data=admin_data)
        self.iris_subset = service_consumption.data.index
        self.geo_data = geo_data
        self.admin_data = admin_data
        self.election_data = election_data
        self.service_consumption = service_consumption
        self.service_subset = ['Facebook', 'Wikipedia', 'LinkedIn', 'Fortnite', 'Netflix']

    def plot(self):
        correlation_data = self._correlation_data()
        correlations_polarization_turnout_entropy, correlations_voter_preferences = self.get_correlations(correlation_data=correlation_data)
        map_facebook_usage, map_polarization = self.get_map_data(correlation_data=correlation_data)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        self.make_map(feature_with_geo_data=map_facebook_usage, ax=axes[0, 0], column='Facebook', label='Facebook Consumption')
        self.make_map(feature_with_geo_data=map_polarization, ax=axes[0, 1], column='Polarization', label='Political Polarization')
        self._make_bar_chart2(bar_chart_data=correlations_polarization_turnout_entropy, ax=axes[1, 0])
        self._make_bar_chart2(bar_chart_data=correlations_voter_preferences, ax=axes[1, 1])
        self._layout(fig=fig, axes=axes)
        fig.savefig(f'{FIGURE_PATH}/elections.pdf', bbox_inches='tight')
        return fig, axes

    def _layout(self, fig, axes):
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        axes[1, 1].legend(loc='upper center', bbox_to_anchor=(-0.1, 1.2), ncols=5, columnspacing=0.5, handletextpad=0.1)
        axes[1, 1].get_yaxis().set_visible(False)

    def get_map_data(self, correlation_data):
        feature = Feature(data=correlation_data['Facebook'].to_frame(), name='Facebook')
        map_facebook_usage = self._feature_with_geo_data_paris(feature=feature)
        feature = Feature(data=correlation_data['Polarization'].to_frame(), name='Polarization')
        map_polarization = self._feature_with_geo_data_paris(feature=feature)
        return map_facebook_usage, map_polarization

    def get_correlations(self, correlation_data):
        correlations = correlation_data.corr()
        correlations_polarization_turnout_entropy = correlations.loc[
            correlations.columns.isin(['Polarization', 'Turnout', 'Entropy']), correlations.index.isin(
                self.service_subset)]
        correlations_voter_preferences = correlations.loc[
            correlations.columns.isin(['Melanchon', 'Le Pen', 'Macron']), correlations.index.isin(
                self.service_subset)]
        return correlations_polarization_turnout_entropy, correlations_voter_preferences
    def make_bar_chart(self, bar_chart_data, ax):
        width = 0.15  # the width of the bars
        multiplier = 0
        x = np.arange(len(bar_chart_data))  # the label locations
        for col in bar_chart_data.columns:
            corr = bar_chart_data[col].values
            offset = width * multiplier
            ax.bar(x=np.arange(len(corr)) + offset, height=corr, width=width, label=col)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Correlation')
        ax.set_xticks(x + 2.5 * width, bar_chart_data.index)
        ax.set_ylim(-0.7, 0.7)
        return ax

    def make_map(self, feature_with_geo_data, ax, column, label):
        feature_with_geo_data.plot(column=column, ax=ax, legend=True, figsize=(10, 10), cmap='plasma',
                                   legend_kwds={"label": label})
        ax.set_axis_off()

    def _feature_with_geo_data_paris(self, feature):
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=self.iris_subset, other_geo_data_types=[GeoDataType.CITY]).set_index(GeoDataType.IRIS.value)
        paris_iris = list(set(iris_geo_data[iris_geo_data[GeoDataType.CITY.value] == 'Paris'].index).intersection(set(feature.data.index)))
        feature_with_geo_data = gpd.GeoDataFrame(feature.data.merge(iris_geo_data, left_index=True, right_index=True))[['geometry', feature.name]].loc[paris_iris]
        feature_with_geo_data[feature.name] = StandardScaler().fit_transform(feature_with_geo_data[feature.name].values.reshape(-1, 1))
        feature_with_geo_data = feature_with_geo_data[feature_with_geo_data[feature.name].abs() < 3]
        return feature_with_geo_data

    def _correlation_data(self):
        consumption_shares = self._consumption_shares()[self.service_subset]
        data = pd.merge(consumption_shares, self._polarization(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._turnout(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._entropy(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._vote_far_right(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._vote_far_left(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._vote_establishment(), left_index=True, right_index=True, how='inner')
        return data

    def _consumption_shares(self):
        consumption_shares = (self.service_consumption.data.T / self.service_consumption.data.sum(axis=1)).T
        consumption_shares = consumption_shares.loc[self.iris_subset].copy()
        return consumption_shares

    def _entropy(self):
        entropy = self.election_feature_calculator.get_election_feature(feature=ElectionFeatureName.ENTROPY, subset=self.iris_subset)
        return entropy.data.rename(columns={'entropy': 'Entropy'})

    def _polarization(self):
        polarization = self.election_feature_calculator.get_election_feature(feature=ElectionFeatureName.POLARIZATION, subset=self.iris_subset)
        return polarization.data.rename(columns={'polarization': 'Polarization'})

    def _turnout(self):
        turnout = self.election_feature_calculator.get_election_feature(feature=ElectionFeatureName.TURNOUT, subset=self.iris_subset)
        return turnout.data.rename(columns={'turnout': 'Turnout'})

    def _vote_far_left(self):
        far_left = self.election_feature_calculator.get_votes_for_party_by_iris(subset=self.iris_subset, party=Party.FRANCE_INSOUMISE)
        return far_left.data.rename(columns={self.election_data.get_list_name(list_number=Party.FRANCE_INSOUMISE.value): 'Melanchon'})

    def _vote_far_right(self):
        far_right = self.election_feature_calculator.get_votes_for_party_by_iris(subset=self.iris_subset, party=Party.LEPEN)
        return far_right.data.rename(columns={self.election_data.get_list_name(list_number=Party.LEPEN.value): 'Le Pen'})

    def _vote_establishment(self):
        establishment = self.election_feature_calculator.get_votes_for_party_by_iris(subset=self.iris_subset, party=Party.RENAISSANCE)
        return establishment.data.rename(columns={self.election_data.get_list_name(list_number=Party.RENAISSANCE.value): 'Macron'})
