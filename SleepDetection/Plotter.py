import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from SleepDetection.Detector import DetectionResult


class DetectionResultPlot:
    def __init__(self, detection_result: DetectionResult):
        self.traffic_time_series_data = detection_result.traffic_time_series_data
        self.daily_component_traffic_time_series_data = detection_result.daily_component_traffic_time_series_data
        self.sleep_change_points = detection_result.sleep_change_points
        self.sleep_scores = detection_result.sleep_scores

    def plot(self):
        figures = [self._make_plot_location(location_id=location_id) for location_id in self.sleep_change_points.columns]
        return figures

    def _make_plot_location(self, location_id):
        fig = go.Figure()
        fig.add_trace(self._get_trace_traffic_location(location_id=location_id))
        fig.add_trace(self._get_trace_traffic_daily_component_location(location_id=location_id))
        fig.add_trace(self._get_trace_sleep_scores(location_id=location_id))
        fig.add_trace(self._get_trace_sleep_change_points_location(location_id=location_id, sleep_state='asleep'))
        fig.add_trace(self._get_trace_sleep_change_points_location(location_id=location_id, sleep_state='awake'))
        fig.update_layout(title_text=f'Sleep patterns for {location_id}', xaxis_rangeslider_visible=True, height=700)
        return fig

    def _get_trace_traffic_location(self, location_id):
        scaled_data = DetectionResultPlot._scale_data(data=self.traffic_time_series_data)
        trace_traffic_location = go.Scatter(x=scaled_data.index, y=scaled_data[location_id], name='Traffic')
        return trace_traffic_location

    def _get_trace_traffic_daily_component_location(self, location_id):
        scaled_data = DetectionResultPlot._scale_data(data=self.daily_component_traffic_time_series_data)
        trace_traffic_daily_component_location = go.Scatter(x=scaled_data.index, y=scaled_data[location_id], name='Daily Component')
        return trace_traffic_daily_component_location

    def _get_trace_sleep_scores(self, location_id):
        scaled_data = DetectionResultPlot._scale_data(data=self.sleep_scores)
        trace_sleep_scores = go.Scatter(x=scaled_data.index, y=scaled_data[location_id], name='Sleep Scores')
        return trace_sleep_scores

    def _get_trace_sleep_change_points_location(self, location_id, sleep_state):
        sleep_change_points_x = self.sleep_change_points[location_id].xs(sleep_state, level="sleep_state").values
        sleep_change_points_y = 0.4 * np.ones(len(sleep_change_points_x))
        trace_sleep_change_points_location = go.Scatter(x=sleep_change_points_x, y=sleep_change_points_y, name=f'{sleep_state}', mode='markers', marker=dict(color='Yellow', symbol='line-ns', size=250, line=dict(width=1, color='Yellow')))
        return trace_sleep_change_points_location

    @staticmethod
    def _scale_data(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns, index=data.index)


class HistogramsSleepPatterns:
    def __init__(self, sleep_change_points: pd.DataFrame):
        self.sleep_change_points = sleep_change_points
        self.sleep_change_points_time = self._get_sleep_change_points_time()
        self.fig = make_subplots(rows=2, cols=1, subplot_titles=("Awake", "Asleep"))

    def plot(self):
        self._make_subplot_sleep_state(sleep_state='awake', row=1, col=1)
        self._make_subplot_sleep_state(sleep_state='asleep', row=2, col=1)
        self.fig.update_layout(title_text="Sleep patterns")
        self.fig.show(renderer="browser")

    def _make_subplot_sleep_state(self, sleep_state, row, col):
        self.fig.add_traces(self._get_traces_sleep_state(sleep_state=sleep_state), rows=row, cols=col)
        self.fig.update_xaxes(title_text="Time", row=row, col=col)
        self.fig.update_yaxes(title_text="Number of occurrences", row=row, col=col)

    def _get_traces_sleep_state(self, sleep_state):
        bins = self._get_bins(sleep_state=sleep_state)
        traces_bar_charts = [self._get_bar_chart_location(location_id=location_id, bins=bins, sleep_state=sleep_state) for location_id in self.sleep_change_points.columns]
        return traces_bar_charts

    def _get_sleep_change_points_time(self):
        return self.sleep_change_points.apply(lambda x: [pd.Timestamp(a).time() for a in x.values], axis=0)

    def _get_bins(self, sleep_state):
        return sorted(np.unique(self.sleep_change_points_time.xs(sleep_state, level="sleep_state").values))

    def _get_bar_chart_location(self, location_id, bins, sleep_state):
        bar_chart_info = self._get_counts_change_point_occurrence_by_time(location_id=location_id, bins=bins, sleep_state=sleep_state)
        x = [str(a) for a in bar_chart_info.index]
        y = list(bar_chart_info.values.flatten())
        trace_bar_chart = go.Bar(x=x, y=y, name=str(location_id), legendgroup=str(location_id))
        return trace_bar_chart

    def _get_counts_change_point_occurrence_by_time(self, location_id, bins, sleep_state):
        sleep_change_point_time_location = self.sleep_change_points_time.xs(sleep_state, level="sleep_state")
        counts_change_point_occurrence_by_time = sleep_change_point_time_location[location_id].groupby(by=sleep_change_point_time_location[location_id]).count()
        counts_change_point_occurrence_by_time_all_bins = pd.DataFrame(index=bins, columns=[location_id])
        counts_change_point_occurrence_by_time_all_bins[location_id] = counts_change_point_occurrence_by_time
        counts_change_point_occurrence_by_time_all_bins.fillna(0, inplace=True)
        return counts_change_point_occurrence_by_time_all_bins
