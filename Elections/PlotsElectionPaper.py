from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import statsmodels.regression.linear_model as statmodels_linear_model

import plotting


data_folder_path = '/Users/anmusso/Desktop/PhD/Projects/Current/NetMob/NetMobData/data/FigureData/GeoDiscontent'
figure_folder_path = '/Users/anmusso/Desktop/PhD/Projects/Current/NetMob/Figures/GeoDiscontent'


def paris_map_plots():
    file_name = 'paris_map.geojson'
    data = gpd.read_file(f'{data_folder_path}/{file_name}')

    plt.rcParams['font.size'] = 30
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'medium'
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw={"width_ratios": [1, 1]})
    axes = axes.flatten()
    columns = ['facebook_share', 'extremist_vote', 'median_income', 'share_high_school_graduates']
    col_name = ['Facebook consumption', 'Radical vote', 'Median income', 'No higher education']
    vmin, vmax = -3, 3
    cmap = 'plasma'
    for i in range(len(columns)):
        plotting.map.simple_normalized_map(ax=axes[i], data=data, column=columns[i], cmap=cmap, legend=False, vmin=vmin, vmax=vmax, outlier_threshold=.995)
        axes[i].set_title(col_name[i])

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([]) # this adds in the colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.1, label='Standardized value')
    plt.show()
    fig.savefig(f'{figure_folder_path}/paris_map.pdf', bbox_inches='tight')


def save_plotly_fig(fig, file_name):
    for i in range(3):
        pio.write_image(fig, f'{figure_folder_path}/{file_name}.png')


def social_media_shares_income_and_extremism():
    file_name = 'social_media_shares_income_and_extremism.csv'
    data = pd.read_csv(f'{data_folder_path}/{file_name}', index_col=0)
    data['median_income'] = data['median_income'].apply(lambda x: np.log(x + 1))
    nbins = 5
    fig = go.Figure()
    traces = plotting.barchart.barchart_mean_values_along_axis(data=data, nbins=nbins, x_axis='median_income', col_label={'social_media_share': 'Social media', 'extremist_vote': 'Radical vote'})
    fig.add_traces(traces)
    fig.update_layout(template='plotly_white', barmode='group', xaxis_title='Standardized Log( median income )', yaxis_title='Standardized value', font=dict(size=30, color="Black", family="Arial"), legend=dict(font=dict(size=30)),
                        width=1000, height=800)
    fig.show()
    save_plotly_fig(fig, 'social_media_shares_income_and_extremism')


def social_media_shares_extremism_scatter_plot():
    file_name = 'social_media_shares_extremism_and_controls.csv'
    data = pd.read_csv(f'{data_folder_path}/{file_name}', index_col=0)
    fig = go.Figure()
    data['social_media_percentage'] = data['social_media_share'] * 100
    trace_scatter, trace_regression_line, reg = plotting.scatter.simple_regression_scatter(data=data, x_axis='social_media_percentage', y_axis='extremist_vote', outlier_threshold=0.99, marker_opacity=0.3, line_width=3)
    fig.add_traces([trace_scatter, trace_regression_line])
    fig.add_annotation(x=0.05, y=0.95, xref='paper', yref='paper', text=f'R<sup>2</sup> = {reg.rsquared:.2f}  Slope = {np.round(reg.params[1], decimals=4)}', showarrow=False, font=dict(size=30, color="Black", family="Arial"))
    fig.update_layout(template='plotly_white', xaxis_title='Percentage social media', yaxis_title='Percentage radical vote', font=dict(size=30, color="Black", family="Arial"), width=1000, height=800)
    fig.show()
    save_plotly_fig(fig, 'social_media_shares_extremism_scatter_plot')


def social_media_shares_with_controls_coefficient_plot():
    file_name = 'social_media_shares_extremism_and_controls.csv'
    data = pd.read_csv(f'{data_folder_path}/{file_name}', index_col=0)
    data['DEC_MED19'] = data['DEC_MED19'].apply(lambda x: np.log(x + 1))
    data['social_media_percentage'] = data['social_media_share'] * 100


    xaxis = 'social_media_percentage'
    yaxis = 'extremist_vote'

    age_vars = ['P19_POP1529', 'P19_POP3044', 'P19_POP4559', 'P19_POP6074', 'P19_POP75P']
    education_vars = ['P19_ACT_DIPLMIN', 'P19_ACT_BAC', 'P19_ACT_SUP2']
    income_vars = ['DEC_MED19']
    unemployment_vars = ['P19_CHOM1524', 'P19_CHOM2554', 'P19_CHOM5564']
    immigration_vars = ['P19_POP_IMM']
    gender_vars = ['P19_POPH']

    barchart = plotting.barchart.barchart_coefficient_with_all_control_combinations(data=data, x_axis=xaxis, y_axis=yaxis,
                                                                                    controls={'age': age_vars, 'education': education_vars, 'gender': gender_vars, 'income': income_vars,
                                                                                              'unemployment': unemployment_vars, 'immigration': immigration_vars},
                                                                                    control_labels={'age': 'A', 'education': 'E', 'gender': 'G', 'income': 'I', 'unemployment': 'U',
                                                                                                    'immigration': 'Im'})
    fig = go.Figure()
    fig.add_trace(barchart)
    fig.show()

    coeff_no_controls, std_err_no_controls = get_coefficient_regression_with_controls(data=data, x_axis=xaxis, y_axis=yaxis, controls=[])
    coeff_income, std_err_income = get_coefficient_regression_with_controls(data=data, x_axis=xaxis, y_axis=yaxis, controls=income_vars)
    coeff_all, std_err_all = get_coefficient_regression_with_controls(data=data, x_axis=xaxis, y_axis=yaxis, controls=income_vars + immigration_vars + education_vars + unemployment_vars + gender_vars + age_vars)

    data = [['No controls', coeff_no_controls, std_err_no_controls],
            ['Model 1', coeff_income, std_err_income],
            ['Model 2', coeff_all, std_err_all]]

    data = pd.DataFrame(data, columns=['controls', 'coefficient', 'std_err'])
    fig = go.Figure()
    barchart_trace = go.Bar(x=data['controls'], y=data['coefficient'], error_y=dict(type='data', array=data['std_err']))
    fig.add_trace(barchart_trace)
    fig.update_layout(template='plotly_white', xaxis_title='Controls', yaxis_title='Coefficient', font=dict(size=30, color="Black", family="Arial"), width=1000, height=800)
    fig.show()
    save_plotly_fig(fig, 'social_media_shares_with_controls_coefficient_plot')


def get_coefficient_regression_with_controls(data: pd.DataFrame, x_axis: str, y_axis: str, controls: List[str]):
    data = data[[x_axis, y_axis] + controls]
    reg = run_multivariable_regression(data=data, y_axis=y_axis)
    return reg.params[x_axis], reg.bse[x_axis]


def run_multivariable_regression(data: pd.DataFrame, y_axis: str) -> statmodels_linear_model.RegressionResults:
    y = data[y_axis]
    X = data.drop(columns=[y_axis])
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results


if __name__ == '__main__':
    paris_map_plots()
    #social_media_shares_income_and_extremism()
    #social_media_shares_extremism_scatter_plot()
    #social_media_shares_with_controls_coefficient_plot()
