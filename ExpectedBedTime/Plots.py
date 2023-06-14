from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from ExpectedBedTime.ExpectedBedTimeCalculator import ExpectedBedTime


class StartBedTimeRobustness:
    def __init__(self, expected_bed_times: List[ExpectedBedTime]):
        self.expected_bed_times = expected_bed_times

    def plot_bed_times_for_different_start_bed_time(self, show_confidence_intervals: bool = True):
        fig = go.Figure()
        for i, expected_bed_time in enumerate(self.expected_bed_times):
            expected_bed_time.plot(fig=fig, trace_name=expected_bed_time.time_index[0], show_plot=False, reference_point=i, show_confidence_intervals=show_confidence_intervals)

        time_index = self.expected_bed_times[0].time_index
        fig.update_yaxes(tickvals=np.arange(1, len(time_index)), ticktext=time_index)
        fig.update_layout(title='Expected bed times', xaxis_title='Iris', yaxis_title='Expected bed time', xaxis_rangeslider_visible=True, font=dict(size=18))
        fig.show(renderer='browser')

    def plot_iris_quantile_membership_counts_for_different_start_bed_times(self, n_quantiles: int = 5):
        iris_quantile_classification = []
        for expected_bed_time in self.expected_bed_times:
            iris_quantile_map_for_bed_time = expected_bed_time.assign_iris_to_quantile(n_quantiles=n_quantiles)
            iris_quantile_map_for_bed_time.rename(columns={'quantile': f'quantile_{expected_bed_time.time_index[0]}'}, inplace=True)
            iris_quantile_classification.append(iris_quantile_map_for_bed_time)

        iris_quantile_classification = pd.concat(iris_quantile_classification, axis=1)
        iris_quantile_membership_counts = iris_quantile_classification.apply(lambda x: x.value_counts(), axis=1)
        iris_quantile_membership_counts.sort_values(by=list(np.arange(1, n_quantiles)), inplace=True)

        fig = px.bar(iris_quantile_membership_counts, x=iris_quantile_membership_counts.index, y=iris_quantile_membership_counts.columns, barmode='stack', labels={'variable': 'Quantile'})
        fig.update_layout(title='Quantile membership counts', xaxis_title='Iris', yaxis_title='Membership count')
        fig.show(renderer='browser')