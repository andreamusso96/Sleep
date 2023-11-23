import pandas as pd
import numpy as np
import plotly.graph_objs as go


def load_data():
    path = '/Users/andrea/Desktop/PhD/Projects/Current/NetMob/Data/BaseData'
    bed_time_index = pd.read_csv(f'{path}/bed_time_index_insee_tile.csv', index_col=0)
    admin_insee_tile = pd.read_csv(f'{path}/admin_data_insee_tile.csv', index_col=0)
    log_mean_income = (np.log2(admin_insee_tile['Ind_snv'] / admin_insee_tile['Ind'])).to_frame('log_mean_income')
    quantiles = np.quantile(log_mean_income, q=[0.3, 0.7])
    categories = pd.cut(log_mean_income['log_mean_income'], bins=[-np.inf, quantiles[0], quantiles[1], np.inf], labels=['low', 'medium', 'high']).to_frame('income_category')
    data = pd.concat([bed_time_index, categories, log_mean_income], axis=1)
    return data


def generate_samples(data: pd.DataFrame, n_samples: int):
    counts_categories = data.groupby('income_category', observed=False).size().to_frame('count')
    n_samples_low = counts_categories.loc['low', 'count']
    n_samples_med = counts_categories.loc['medium', 'count']
    n_samples_high = counts_categories.loc['high', 'count']

    samples = []
    for i in range(n_samples):
        sample = generate_sample(data['bed_time_index'].values, n_samples_low, n_samples_med, n_samples_high)
        samples.append(sample)

    samples = pd.concat(samples, axis=0).reset_index(drop=True)
    return samples


def generate_sample(x, n_samples_low, n_samples_med, n_samples_high):
    low_samples = np.random.choice(x, size=n_samples_low, replace=True)
    med_samples = np.random.choice(x, size=n_samples_med, replace=True)
    high_samples = np.random.choice(x, size=n_samples_high, replace=True)
    mean_low = np.mean(low_samples)
    mean_med = np.mean(med_samples)
    mean_high = np.mean(high_samples)
    diff_mean_low_med = mean_low - mean_med
    diff_mean_med_high = mean_med - mean_high
    diff_mean_low_high = mean_low - mean_high
    sample = pd.DataFrame({'diff_mean_low_med': diff_mean_low_med, 'diff_mean_med_high': diff_mean_med_high, 'diff_mean_low_high': diff_mean_low_high}, index=[0])
    return sample


def violin_plot(samples, real_values):
    fig = go.Figure()
    names = ['low - medium', 'medium - high', 'low - high']
    for i, col in enumerate(samples.columns):
        fig.add_trace(go.Violin(y=samples[col], name=names[i], box_visible=True, meanline_visible=True))
    fig.add_trace(go.Scatter(x=names, y=real_values.values[0], mode='markers', name='real values', marker=dict(color='black', size=10, symbol='x')))
    fig.update_layout(title='Confidence intervals for the difference of means', xaxis_title='Difference of means', yaxis_title='Value', font=dict(size=30, color='black'), template='plotly_white')
    fig.show(renderer='browser')
    return fig




if __name__ == '__main__':
    d = load_data()
    s = generate_samples(data=d, n_samples=10000)
    real_mean_by_income_category = d.groupby('income_category', observed=False).agg({'bed_time_index': 'mean'})
    real_mean_low, real_mean_med, real_mean_high = real_mean_by_income_category.loc['low', 'bed_time_index'], real_mean_by_income_category.loc['medium', 'bed_time_index'], real_mean_by_income_category.loc['high', 'bed_time_index']
    real_diff_means = pd.DataFrame({'diff_mean_low_med': real_mean_low - real_mean_med, 'diff_mean_med_high': real_mean_med - real_mean_high, 'diff_mean_low_high': real_mean_low - real_mean_high}, index=[0])
    fig = violin_plot(samples=s, real_values=real_diff_means)
    import plotly.io as pio
    pio.write_html(fig, file='/Users/andrea/Desktop/PhD/Presentations/HTMLPresentations/reveal.js/assets/2023-10-06-Sleep/confidence_intervals.html', auto_open=True)

