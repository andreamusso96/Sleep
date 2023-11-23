import pandas as pd
import numpy as np
import plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

data_folder_path = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/BaseData'
figure_folder_path = '/Users/andrea/Desktop/PhD/Presentations/HTMLPresentations/reveal.js/assets/2023-10-06-Sleep/'


def plot_income_vs_night_screen_index():
    night_screen_index = pd.read_csv(f'{data_folder_path}/bed_time_index_insee_tile.csv', index_col=0)
    admin_data = pd.read_csv(f'{data_folder_path}/admin_data_insee_tile.csv', index_col=0)
    income = (admin_data['Ind_snv'] / admin_data['Ind']).to_frame('income')
    log_income = np.log2(income['income']).to_frame('log_income')
    data = pd.concat([night_screen_index, log_income], axis=1)
    data = data[['log_income', 'bed_time_index']].dropna()

    x_axis = 'log_income'
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03)
    heatmap, line_heatmap = plotting.heatmap.simple_heat_map_with_mean_value(data=data, x_axis=x_axis, y_axis='bed_time_index', outlier_threshold=0.99, line_width=5, bins=10, line_color='red', showscale=False)
    scatter, line_scatter, reg_res = plotting.scatter.simple_regression_scatter(data=data, x_axis=x_axis, y_axis='bed_time_index', line_width=5, line_color='red', marker_opacity=0.2)
    fig.add_traces([heatmap, line_heatmap], rows=1, cols=1)
    fig.add_traces([scatter, line_scatter], rows=1, cols=2)
    fig.add_annotation(x=0.95, y=0.95, text=f'Slope: {reg_res.params[1]:.2f}', showarrow=False, xref='paper', yref='paper', font=dict(size=30))
    fig.update_xaxes(title='Log2 Median Income')
    fig.update_yaxes(title='Night Screen Index')
    fig.update_layout(template='plotly_white', font=dict(size=30, color='black'))
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/income_vs_night_screen_index.html', auto_open=True)


def plot_noise_vs_night_screen_index():
    night_screen_index = pd.read_csv(f'{data_folder_path}/bed_time_index_insee_tile.csv', index_col=0)
    noise = pd.read_csv(f'{data_folder_path}/noise_insee_tile.csv', index_col=0)
    data = pd.concat([night_screen_index, noise], axis=1)
    data = data[['noise_estimate', 'bed_time_index']].dropna()

    x_axis = 'noise_estimate'
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03)
    heatmap, line_heatmap = plotting.heatmap.simple_heat_map_with_mean_value(data=data, x_axis=x_axis, y_axis='bed_time_index', outlier_threshold=0.99, line_width=5, bins=10, line_color='red', showscale=False)
    scatter, line_scatter, reg_res = plotting.scatter.simple_regression_scatter(data=data, x_axis=x_axis, y_axis='bed_time_index', line_width=5, line_color='red', marker_opacity=0.2)
    fig.add_traces([heatmap, line_heatmap], rows=1, cols=1)
    fig.add_traces([scatter, line_scatter], rows=1, cols=2)
    fig.add_annotation(x=0.95, y=0.95, text=f'Slope: {reg_res.params[1]:.2f}', showarrow=False, xref='paper', yref='paper', font=dict(size=30))
    fig.update_xaxes(title='Noise (dB)')
    fig.update_yaxes(title='Night Screen Index')
    fig.update_layout(template='plotly_white', font=dict(size=30, color='black'))
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/noise_vs_night_screen_index.html', auto_open=True)


def plot_home_work_trip_duration_vs_night_screen_index():
    bed_time_index_and_avg_duration_income = pd.read_csv(f'{data_folder_path}/bed_time_index_avg_work_to_home_trip_duration_income_commune.csv', index_col=0)
    bed_time_index_and_avg_duration_income['log_2_income'] = np.log2(bed_time_index_and_avg_duration_income['income'])
    bed_time_index_and_avg_duration_income['total_duration_minutes'] = bed_time_index_and_avg_duration_income['total_duration_seconds'] / 60
    data = bed_time_index_and_avg_duration_income[['total_duration_minutes', 'bed_time_index', 'log_2_income']].dropna()

    y_axis = 'total_duration_minutes'
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03)
    scatter_night_screen_index, line_scatter_night_screen_index, reg_res_night_screen_index = plotting.scatter.simple_regression_scatter(data=data, x_axis='bed_time_index', y_axis=y_axis, line_width=5, line_color='red', marker_opacity=0.5)
    scatter_income, line_scatter_income, reg_res_income = plotting.scatter.simple_regression_scatter(data=data, x_axis='log_2_income', y_axis=y_axis, line_width=5, line_color='red', marker_opacity=0.5)
    fig.add_traces([scatter_night_screen_index, line_scatter_night_screen_index], rows=1, cols=1)
    fig.add_traces([scatter_income, line_scatter_income], rows=1, cols=2)
    fig.add_annotation(x=0.2, y=0.95, text=f'Slope: {reg_res_night_screen_index.params[1]:.2f} (p-value {reg_res_night_screen_index.pvalues[1]:.2f})', showarrow=False, xref='paper', yref='paper', font=dict(size=30))
    fig.add_annotation(x=1, y=0.95, text=f'Slope: {reg_res_income.params[1]:.2f} (p-value {reg_res_income.pvalues[1]:.2f})', showarrow=False, xref='paper', yref='paper', font=dict(size=30))
    fig.update_yaxes(title='Average Work to Home Trip Duration (minutes)')
    fig.update_xaxes(title='Night Screen Index', row=1, col=1)
    fig.update_xaxes(title='Log2 Median Income', row=1, col=2)
    fig.update_layout(template='plotly_white', font=dict(size=30, color='black'))
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/home_work_trip_duration_vs_night_screen_index.html', auto_open=True)


def plot_amenities_vs_night_screen_index():
    amenity_counts = pd.read_csv(f'{data_folder_path}/amenity_counts_insee_tile.csv', index_col=0)
    night_screen_index = pd.read_csv(f'{data_folder_path}/bed_time_index_insee_tile.csv', index_col=0)
    data = pd.merge(amenity_counts, night_screen_index, left_index=True, right_index=True)
    data = data.dropna()
    data['log_amenity_count'] = np.log2(data['amenity_count'])
    data['log_frequently_visited_amenity_count'] = np.log2(data['frequently_visited_amenity_count'])
    data = data[['bed_time_index', 'log_amenity_count', 'log_frequently_visited_amenity_count']].dropna()

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03)
    heatmap_amenity, line_heatmap_amenity = plotting.heatmap.simple_heat_map_with_mean_value(data=data, x_axis='log_amenity_count', y_axis='bed_time_index', outlier_threshold=0.99, line_width=5, bins=10, line_color='red', showscale=False)
    heatmap_fv_amenity, line_heatmap_fv_amenity = plotting.heatmap.simple_heat_map_with_mean_value(data=data, x_axis='log_frequently_visited_amenity_count', y_axis='bed_time_index', outlier_threshold=0.99, line_width=5, bins=10, line_color='red', showscale=False)
    fig.add_traces([heatmap_amenity, line_heatmap_amenity], rows=1, cols=1)
    fig.add_traces([heatmap_fv_amenity, line_heatmap_fv_amenity], rows=1, cols=2)
    fig.update_xaxes(title='Log2 Number of Amenities', row=1, col=1)
    fig.update_xaxes(title='Log 2 Number of Frequently Visited Amenities', row=1, col=2)
    fig.update_yaxes(title='Night Screen Index')
    fig.update_layout(template='plotly_white', font=dict(size=30, color='black'))
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/amenities_vs_night_screen_index.html', auto_open=True)


def plot_rca_app_usage_by_income_bracket(services = None):
    mobile_usage_insee_tile = pd.read_csv(f'{data_folder_path}/app_consumption_insee_tile.csv', index_col=0)
    services = list(mobile_usage_insee_tile.columns) if services is None else services
    mobile_usage_insee_tile = mobile_usage_insee_tile[services].copy()
    categories = _load_income_and_split_into_categories(quantiles=[0.3, 0.7], labels=['low', 'medium', 'high'])
    mobile_usage_income_category = mobile_usage_insee_tile.merge(categories, left_index=True, right_index=True)
    mobile_usage_income_category = mobile_usage_income_category.groupby('income_category', observed=False).sum()
    mobile_usage_income_category_rca = compute_rca(df=mobile_usage_income_category)
    mobile_usage_income_category_rca = mobile_usage_income_category_rca.T.sort_values(by='low', ascending=False).T

    fig = go.Figure()
    heatmap = go.Heatmap(
        z=mobile_usage_income_category_rca.values,
        x=mobile_usage_income_category_rca.columns,
        y=mobile_usage_income_category_rca.index,
        zmid=1,
        colorscale='RdBu',
        colorbar=dict(
            title='RCA',
            titleside='right'
        ))
    fig.add_trace(heatmap)
    fig.update_layout(
        title='RCA of mobile app usage by income category',
        xaxis_title='Mobile app',
        yaxis_title='Income category',
        template='plotly_white',
        font=dict(size=18, color='black'))
    fig.show(renderer='browser')
    pio.write_html(fig, file=f'{figure_folder_path}/rca_app_usage_by_income_bracket.html', auto_open=True)


def compute_rca(df):
    numerator = df.div(df.sum(axis=1), axis=0)
    denominator = df.sum(axis=0) / df.sum().sum()
    return numerator.div(denominator, axis=1)


def _load_income_and_split_into_categories(quantiles, labels):
    income = pd.read_csv(f'{data_folder_path}/admin_data_insee_tile.csv', index_col=0)
    income['mean_income'] = income['Ind_snv'] / income['Ind']
    log2_income = np.log2(income['mean_income'])
    qs = list(np.quantile(log2_income.values, q=quantiles))
    bins = [-np.inf] + qs + [np.inf]
    categories = pd.cut(log2_income, bins=bins, labels=labels).to_frame('income_category')
    return categories


def get_selected_services():
    import mobile_traffic as mt
    s = mt.Service
    selected_services = [s.PLAYSTATION, s.SNAPCHAT, s.FORTNITE, s.EA_GAMES, s.WHATSAPP, s.PERISCOPE, s.WEB_STREAMING, s.WEB_ADULT, s.ORANGE_TV, s.WEB_E_COMMERCE, s.TELEGRAM, s.FACEBOOK_MESSENGER, s.FACEBOOK_LIVE, s.WEB_GAMES, s.FACEBOOK, s.CLASH_OF_CLANS, s.PINTEREST, s.WEB_FINANCE, s.TWITTER, s.YOUTUBE, s.TWITCH, s.NETFLIX, s.DAILYMOTION, s.SKYPE, s.MOLOTOV, s.WEB_CLOTHES, s.GOOGLE_DOCS, s.INSTAGRAM, s.DEEZER, s.GOOGLE_DRIVE, s.WEB_FOOD, s.SOUNDCLOUD, s.TOR, s.POKEMON_GO, s.SPOTIFY, s.GOOGLE_MAIL, s.GOOGLE_MEET, s.TEAMVIEWER, s.WIKIPEDIA, s.YAHOO, s.MICROSOFT_MAIL, s.MICROSOFT_OFFICE, s.LINKEDIN]
    selected_services = [s.value for s in selected_services]
    return selected_services


if __name__ == '__main__':
    plot_rca_app_usage_by_income_bracket(services=get_selected_services())