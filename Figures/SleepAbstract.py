import pandas as pd
import numpy as np
import plotly.express as px

from DataInterface.GeoDataInterface import GeoData, GeoDataType
from DataInterface.AdminDataInterface import AdminData
from FeatureExtraction.Feature import Feature
from FeatureExtraction.IrisFeatureCalculator import IrisFeatureCalculator
from FeatureSelection.Controls import NoControl, AgeControl, EducationControl, IncomeControl
from FeatureSelection.Regression import Regression
from FeatureSelection.MultiRegressionPlotter import MultiRegressionPlotLayoutParameters, RegressionSubplotScatter, RegressionSubplotHeatmap, MultiRegressionPlot
from config import FIGURE_PATH


def log(df: pd.DataFrame):
    vals = df.values
    vals = np.log(vals, out=np.zeros_like(vals), where=(vals != 0))
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


class SleepAbstractRegressionData:
    def __init__(self, geo_data: GeoData, session_expectation: Feature, admin_data: AdminData):
        self.geo_data = geo_data
        self.session_expectation = session_expectation
        self.admin_data = admin_data
        self.iris_feature_calculator = IrisFeatureCalculator(geo_data=geo_data, admin_data=admin_data)
        self.iris_subset = self.session_expectation.data.index

    def get_regression_data(self):
        data = pd.merge(log(self._insomnia_index()), log(self._density()), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, log(self._amenity_index()), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, log(self._transportation_index()), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, log(self._age_controls()), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, log(self._income_controls()), left_index=True, right_index=True, how='inner')
        data = pd.merge(data, log(self._education_controls()), left_index=True, right_index=True, how='inner')
        data = data[data != 0]
        data.dropna(inplace=True)
        return data

    def _amenity_index(self):
        services = ['A1', 'A2', 'A5']
        shops = ['B1', 'B2', 'B3']
        schools = ['C1', 'C2', 'C3', 'C4' 'C5', 'C6']
        health_care = ['D1', 'D2', 'D3', 'D4', 'D5']
        free_time = ['F1', 'F2', 'F3']
        tourism = ['G1']
        coarsened_equip = True
        equip_names = [c for c in self.admin_data.get_admin_data(coarsened_equip=coarsened_equip).columns if
                       c.startswith('EQUIP_')]
        equip_names = [c for c in equip_names if
                       c.split('_')[1] in services + shops + schools + health_care + free_time + tourism]
        amenity_index = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=equip_names,
                                                                 coarsened_equip=coarsened_equip).sum(axis=1).to_frame('Amenity Index')
        return amenity_index

    def _density(self):
        density = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=['P19_POP']).rename(columns={'P19_POP': 'Density'})
        return density

    def _transportation_index(self):
        transportation = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=['EQUIP_E1'], coarsened_equip=True).rename(columns={'EQUIP_E1': 'Transportation Index'})
        return transportation

    def _insomnia_index(self):
        insomnia_index = self.session_expectation.data.rename(columns={'session_expectation': 'Insomnia Index'})['Insomnia Index'].to_frame()
        return insomnia_index

    def _age_controls(self):
        age = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=AgeControl().var_names)
        return age

    def _income_controls(self):
        income = self.admin_data.get_admin_data(subset=self.iris_subset)[IncomeControl().var_names]
        return income

    def _education_controls(self):
        education = self.iris_feature_calculator.var_density(subset=self.iris_subset, var_names=EducationControl().var_names)
        return education


class SleepAbstractFigure:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.layout_parameters = self._get_layout_parameters()
        self.save_path = f'{FIGURE_PATH}/insomnia_index_scatter.pdf'

    def show(self, save: bool = False, scatter: bool = True):
        regression_subplot_class = RegressionSubplotScatter if scatter else RegressionSubplotHeatmap
        regression_subplots = self._get_regression_subplots(regression_subplot_class=regression_subplot_class)
        multi_regression_plot = MultiRegressionPlot(regression_subplots=regression_subplots, layout_parameters=self.layout_parameters)
        fig = multi_regression_plot.plot()
        if save:
            fig.write_image(self.save_path)

    def _get_all_controls(self):
        all_controls = [NoControl(), AgeControl(), AgeControl().join(EducationControl()), AgeControl().join(IncomeControl()), AgeControl().join(EducationControl()).join(IncomeControl())]
        return all_controls

    def _get_regressions(self):
        treatments = ['Density', 'Amenity Index', 'Transportation Index', 'DEC_MED19', 'P19_ACT_DIPLMIN']
        outcome = 'Insomnia Index'
        regressions = []
        for treatment in treatments:
            regression = Regression(data=self.data, treatment=treatment, outcome=outcome)
            regressions.append(regression)
        return regressions

    def _get_regression_subplots(self, regression_subplot_class):
        map_treatment_to_xtitle = {'Density': 'log(Density)', 'Transportation Index': 'log(Transportation Index)', 'Amenity Index': 'log(Amenity Index)', 'DEC_MED19': 'log(Median Income)', 'P19_ACT_DIPLMIN': 'log(Density HS Graduates)'}
        map_outcome_to_ytitle = {'Insomnia Index': 'log(Insomnia Index)'}
        controls = self._get_all_controls()
        controls_to_skip = [[3], [3], [3], [3, 4], [2, 4]]
        regressions = self._get_regressions()
        regression_subplots = []
        for i, regression in enumerate(regressions):
            controls_ = [x for j, x in enumerate(controls) if j not in controls_to_skip[i]]
            regression_subplot = regression_subplot_class(regression=regression, controls=controls_, xtitle=map_treatment_to_xtitle[regression.treatment], ytitle=map_outcome_to_ytitle[regression.outcome])
            regression_subplots.append(regression_subplot)

        return regression_subplots

    def _get_layout_parameters(self):
        controls = self._get_all_controls()
        inset_bar_legend_names = ['No Controls', 'Age', 'Age + Edu', 'Age + Inc', 'Age + Edu + Inc']
        colors = px.colors.qualitative.Plotly[:len(inset_bar_legend_names)]
        inset_bar_colors = {control.name: colors[i] for i, control in enumerate(controls)}
        layout_params = MultiRegressionPlotLayoutParameters(nrows=2, ncols=3, width=1200, height=800, vertical_spacing=0.1,
                                                            horizontal_spacing=0.03, font_size=21, line_width=3,
                                                            template='plotly_white',
                                                            inset_font_size=18, inset_line_width=2,
                                                            inset_bar_legend_names=inset_bar_legend_names,
                                                            inset_bar_colors=inset_bar_colors,
                                                            inset_size_x=0.1, inset_size_y=0.1, inset_shift_x=0.01,
                                                            inset_shift_y=0.02, legend_x=0.1, legend_y=1.01, skip=[(2, 2)])
        return layout_params