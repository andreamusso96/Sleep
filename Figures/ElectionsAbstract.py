import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from DataInterface.GeoDataInterface import GeoData, GeoDataType
from DataInterface.AdminDataInterface import AdminData
from DataInterface.ElectionDataInterface import ElectionData, ElectionFeatureName, Party
from FeatureExtraction.ElectionFeatureCalculator import ElectionFeatureCalculator
from FeatureExtraction.ServiceConsumptionFeatureCalculator import ServiceConsumptionFeatureCalculator
from FeatureExtraction.ServiceConsumption.ServiceConsumption import ServiceConsumption
from FeatureExtraction.IrisFeatureCalculator import IrisFeatureCalculator
from FeatureExtraction.Feature import Feature
from FeatureSelection.Controls import NoControl, AgeControl, IncomeControl, EducationControl
from FeatureSelection.MultiRegressionPlotter import MultiRegressionPlotLayoutParameters, RegressionSubplotScatter, RegressionSubplotHeatmap, MultiRegressionPlot
from FeatureSelection.Regression import Regression
from Utils import City
from config import FIGURE_PATH


def log(df: pd.DataFrame):
    vals = df.values
    vals = np.log(vals, out=np.zeros_like(vals), where=(vals != 0))
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


class CityMap:
    def __init__(self, feature: Feature, geo_data: GeoData, city: City):
        self.feature = feature
        self.geo_data = geo_data
        self.city = city
        self._normalized_feature_with_geo_data = self._normalized_feature_with_geo_data()

    def make_map(self, save: bool = False):
        fig, ax = plt.subplots(figsize=(10, 10))
        self._normalized_feature_with_geo_data.plot(column=self.feature.name, ax=ax, legend=False, figsize=(10, 10), cmap='Greys')
        ax.set_axis_off()
        fig.savefig(f'{FIGURE_PATH}/map_{self.feature.name}_{self.city.value}.pdf', bbox_inches='tight')

    def _normalized_feature_with_geo_data(self) -> gpd.GeoDataFrame:
        iris_subset_feature = self.feature.data.index
        iris_geo_data = self.geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=iris_subset_feature, other_geo_data_types=[GeoDataType.CITY]).set_index(GeoDataType.IRIS.value)
        iris_subset = list(set(iris_geo_data[iris_geo_data[GeoDataType.CITY.value] == self.city.value].index).intersection(set(iris_subset_feature)))
        feature_with_geo_data = gpd.GeoDataFrame(self.feature.data.merge(iris_geo_data, left_index=True, right_index=True))[['geometry', self.feature.name]].loc[iris_subset]
        feature_with_geo_data[self.feature.name] = StandardScaler().fit_transform(feature_with_geo_data[self.feature.name].values.reshape(-1, 1))
        feature_with_geo_data = feature_with_geo_data[feature_with_geo_data[self.feature.name].abs() < 3]
        return feature_with_geo_data


class ElectionAbstractRegressionData:
    def __init__(self, geo_data: GeoData, admin_data: AdminData, election_data: ElectionData, service_consumption: ServiceConsumption):
        self.election_feature_calculator = ElectionFeatureCalculator(election_data=election_data, geo_data=geo_data)
        self.service_consumption_feature_calculator = ServiceConsumptionFeatureCalculator(service_consumption=service_consumption, admin_data=admin_data)
        self.iris_feature_calculator = IrisFeatureCalculator(geo_data=geo_data, admin_data=admin_data)
        self.iris_subset = service_consumption.data.index
        self.geo_data = geo_data
        self.admin_data = admin_data
        self.election_data = election_data
        self.service_consumption = service_consumption
        self.service_subset = ['Facebook', 'Wikipedia', 'LinkedIn', 'Fortnite', 'Netflix']

    def get_regression_data(self):
        consumption_shares = self._consumption_shares()[self.service_subset].copy()
        data = pd.merge(consumption_shares, self._polarization(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._turnout(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._entropy(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._vote_far_right(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._vote_far_left(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._vote_establishment(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._age_controls(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._income_controls(), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, self._education_controls(), left_index=True, right_index=True, how='inner')
        data = log(data)
        data.dropna(inplace=True)
        return data

    def _consumption_shares(self):
        consumption_shares = (self.service_consumption.data.T / self.service_consumption.data.sum(axis=1)).T
        consumption_shares = consumption_shares.loc[self.iris_subset].copy()
        return consumption_shares

    def _entropy(self):
        entropy = self.election_feature_calculator.get_election_feature(feature=ElectionFeatureName.ENTROPY,
                                                                        subset=self.iris_subset).data
        return entropy

    def _polarization(self):
        polarization = self.election_feature_calculator.get_election_feature(feature=ElectionFeatureName.POLARIZATION,
                                                                             subset=self.iris_subset).data
        return polarization

    def _turnout(self):
        turnout = self.election_feature_calculator.get_election_feature(feature=ElectionFeatureName.TURNOUT,
                                                                        subset=self.iris_subset).data
        return turnout

    def _vote_far_left(self):
        far_left = self.election_feature_calculator.get_votes_for_party_by_iris(subset=self.iris_subset,
                                                                                party=Party.FRANCE_INSOUMISE).data
        return far_left

    def _vote_far_right(self):
        far_right = self.election_feature_calculator.get_votes_for_party_by_iris(subset=self.iris_subset,
                                                                                 party=Party.LEPEN).data
        return far_right

    def _vote_establishment(self):
        establishment = self.election_feature_calculator.get_votes_for_party_by_iris(subset=self.iris_subset,
                                                                                     party=Party.RENAISSANCE).data
        return establishment

    def _age_controls(self):
        age = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=AgeControl().var_names)
        return age

    def _income_controls(self):
        income = self.admin_data.get_admin_data(subset=self.iris_subset)[IncomeControl().var_names]
        return income

    def _education_controls(self):
        education = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=EducationControl().var_names)
        return education


class ElectionAbstractFigure:
    def __init__(self, data: pd.DataFrame, scatter: bool = True):
        self.data = data
        self.layout_parameters = self._get_layout_parameters()
        self.scatter = scatter
        self.save_path = f'{FIGURE_PATH}/election_correlations_scatter.pdf'

    def show(self, save: bool = False, scatter: bool = True):
        regression_subplot_class = RegressionSubplotScatter if scatter else RegressionSubplotHeatmap
        regression_subplots = self._get_regression_subplots(regression_subplot_class=regression_subplot_class)
        multi_regression_plot = MultiRegressionPlot(regression_subplots=regression_subplots, layout_parameters=self.layout_parameters)
        fig = multi_regression_plot.plot()
        if save:
            fig.write_image(self.save_path)
        return fig

    def _get_base_controls(self):
        controls = [NoControl(), AgeControl(), AgeControl().join(EducationControl()),
                    AgeControl().join(EducationControl()).join(IncomeControl())]
        return controls

    def _get_regressions(self):
        treatments = ['Facebook', 'Wikipedia', 'LinkedIn', 'Fortnite']
        outcomes = ['polarization'] # ['entropy', 'polarization', 'turnout', Party.FRANCE_INSOUMISE.value, Party.LEPEN.value, Party.RENAISSANCE.value]
        regressions = []
        for outcome in outcomes:
            for treatment in treatments:
                regression = Regression(data=self.data, treatment=treatment, outcome=outcome)
                regressions.append(regression)
        return regressions

    def _get_regression_subplots(self, regression_subplot_class):
        map_treatment_to_xtitle = {'Facebook': 'log(Facebook Share)', 'Wikipedia': 'log(Wikipedia Share)', 'LinkedIn': 'log(LinkedIn Share)', 'Fortnite': 'log(Fortnite Share)', 'Netflix': 'log(Netflix Share)'}
        map_outcome_to_ytitle = {'entropy': 'log(Entropy)', 'polarization': 'log(Polarization)', 'turnout': 'log(Turnout)', Party.FRANCE_INSOUMISE.value: 'log(Votes Far Left)', Party.LEPEN.value: 'log(Votes Far Right)', Party.RENAISSANCE.value: 'log(Votes Establishment)'}
        controls = self._get_base_controls()
        regression_subplots = []
        regressions = self._get_regressions()
        for regression in regressions:
            regression_subplot = regression_subplot_class(regression=regression, controls=controls, xtitle=map_treatment_to_xtitle[regression.treatment], ytitle=map_outcome_to_ytitle[regression.outcome])
            regression_subplots.append(regression_subplot)
        return regression_subplots

    def _get_layout_parameters(self):
        controls = self._get_base_controls()
        inset_bar_legend_names = ['No Controls', 'Age', 'Age + Edu', 'Age + Edu + Inc']
        colors = px.colors.qualitative.Plotly[:len(inset_bar_legend_names)]
        inset_bar_colors = {control.name: colors[i] for i, control in enumerate(controls)}
        nrows, ncols = 2, 3
        layout_params = MultiRegressionPlotLayoutParameters(nrows=nrows, ncols=ncols, width=ncols * 400, height=nrows * 400, vertical_spacing=0.1,
                                                            horizontal_spacing=0.02, font_size=21, line_width=3,
                                                            template='plotly_white',
                                                            inset_font_size=18, inset_line_width=2,
                                                            inset_bar_legend_names=inset_bar_legend_names,
                                                            inset_bar_colors=inset_bar_colors,
                                                            inset_size_x=0.1, inset_size_y=0.1, inset_shift_x=0.01,
                                                            inset_shift_y=0.03, legend_x=-0.05, legend_y=1.01, skip=[(1, 3), (2, 3)])
        return layout_params


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
