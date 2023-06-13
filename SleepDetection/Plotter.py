import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xarray as xr

from SleepDetection.Detector import DetectionResult
from Utils import TrafficDataDimensions, Service, TrafficType, AggregationLevel, Indexing


class DetectionResultPlot:
    def __init__(self, detection_result: DetectionResult):
        self.traffic_time_series_data = detection_result.traffic_time_series_data
        self.daily_component_traffic_time_series_data = detection_result.daily_component_traffic_time_series_data
        self.sleep_change_points = detection_result.sleep_change_points
        self.sleep_scores = detection_result.sleep_scores
        self.location_ids = [c for c in self.sleep_change_points.columns if not c.startswith('unc_')]

    def plot(self):
        figures = [self._make_plot_location(location_id=location_id) for location_id in self.location_ids]
        return figures

    def _make_plot_location(self, location_id):
        fig = go.Figure()
        fig.add_trace(self._get_trace_traffic_location(location_id=location_id))
        fig.add_trace(self._get_trace_traffic_daily_component_location(location_id=location_id))
        fig.add_trace(self._get_trace_sleep_scores(location_id=location_id))
        fig.add_trace(self._get_trace_sleep_change_points_location(location_id=location_id, sleep_state='asleep'))
        fig.add_trace(self._get_trace_sleep_change_points_location(location_id=location_id, sleep_state='awake'))
        fig = DetectionResultPlot._set_layout(fig=fig, location_id=location_id)
        return fig

    @staticmethod
    def _set_layout(fig, location_id):
        font = dict(size=18)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Traffic')
        fig.update_layout(title_text=f'Sleep patterns for location {location_id}', xaxis_rangeslider_visible=True,
                          height=700, font=font)
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
        sleep_change_points_uncertainty = 1 + 3*self.sleep_change_points[f'unc_{location_id}'].xs(sleep_state, level="sleep_state").values
        trace_sleep_change_points_location = go.Scatter(x=sleep_change_points_x, y=sleep_change_points_y, name=f'{sleep_state}', mode='markers', marker=dict(color='Yellow', symbol='line-ns', size=250, line=dict(width=sleep_change_points_uncertainty)))
        return trace_sleep_change_points_location

    @staticmethod
    def _scale_data(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns, index=data.index)


class HistogramsSleepPatterns:
    def __init__(self, sleep_change_points: pd.DataFrame):
        self.sleep_change_points = sleep_change_points[[c for c in sleep_change_points.columns if c.startswith('unc_')]]
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


class UsersVsTraffic:
    def __init__(self, xar_city_users: xr.DataArray, xar_city_traffic: xr.DataArray):
        self.xar_city_users = xar_city_users
        self.xar_city_traffic = xar_city_traffic
        self.fig = go.Figure()

    def plot(self):
        xar_traffic_ul_dl = self.xar_city_traffic.sel(service=Service.get_services(traffic_type=TrafficType.USERS, return_values=True)).sum(dim=TrafficDataDimensions.SERVICE.value).sum(dim=AggregationLevel.IRIS.value)
        xar_traffic_users = self.xar_city_users.sum(dim=TrafficDataDimensions.SERVICE.value).sum(dim=AggregationLevel.IRIS.value)
        self.plot_location(xar=xar_traffic_ul_dl, name=TrafficType.UL_AND_DL.value)
        self.plot_location(xar=xar_traffic_users, name=TrafficType.USERS.value)
        self.fig.update_layout(title_text="Users vs Traffic")
        self._layout()
        self.fig.show(renderer="browser")

    def plot_location(self, xar: xr.DataArray, name: str):
        service_time_series = Indexing.day_time_to_datetime_index(xar=xar).T.to_pandas()
        rescaled_service_time_series = MinMaxScaler().fit_transform(service_time_series.values.reshape(-1, 1)).flatten()
        trace_service = go.Scatter(x=service_time_series.index, y=rescaled_service_time_series, mode='lines', name=f'{name}')
        self.fig.add_trace(trace_service)

    def _layout(self):
        self.fig.update_layout(title_text="Users vs Traffic", xaxis_rangeslider_visible=True, font=dict(size=18))
        self.fig.update_xaxes(title_text="Time")
        self.fig.update_yaxes(title_text="Number of users / Traffic")


class SeasonPlot:
    def __init__(self, xar_city, n_days_season):
        self.xar_city = xar_city
        self.users = self.xar_city.sel(service=Service.get_services(traffic_type=TrafficType.USERS, return_values=True)).sum(dim=TrafficDataDimensions.SERVICE.value).sum(dim=AggregationLevel.IRIS.value)
        self.n_days_season = n_days_season
        self.n_days_total = self.users.sizes[TrafficDataDimensions.DAY.value]
        self.fig = go.Figure()

    def plot(self):
        seasons = self._split_data_into_seasons()
        for season_id in seasons.columns:
            self._scatter_plot(x=seasons.index, y=seasons[season_id].values.flatten(), name=season_id, opacity=0.2)

        self._scatter_plot(x=seasons.index, y=seasons.mean(axis=1).values.flatten(), name="mean", opacity=1)
        self.fig.update_layout(title_text=f"Season length {self.n_days_season} days")
        self.fig.show(renderer="browser")

    def _scatter_plot(self, x, y, name, opacity):
        trace = go.Scatter(x=x, y=y, mode='lines', line=dict(color=f'rgba(255, 0, 0, {opacity})'))
        self.fig.add_trace(trace)

    def _split_data_into_seasons(self):
        seasons = [self.users.isel(day=list(range(i*self.n_days_season, (i+1)*self.n_days_season))) for i in range(self.n_days_total//self.n_days_season)]
        seasons = self._reformat_seasons(seasons)
        return seasons

    @staticmethod
    def _reformat_seasons(seasons):
        flat_seasons = [season.stack(datetime=[TrafficDataDimensions.DAY.value, TrafficDataDimensions.TIME.value]).T.to_pandas() for season in seasons]
        time_index = flat_seasons[0].index.get_level_values(0) + flat_seasons[0].index.get_level_values(1)
        scaled_seasons = [MinMaxScaler().fit_transform(season.values.reshape(-1, 1)).flatten() for season in
                          flat_seasons]
        season_array = np.stack(scaled_seasons, axis=0).T
        season_df = pd.DataFrame(season_array, index=time_index, columns=list(range(len(seasons))))
        return season_df


class UsersAndWeather:
    def __init__(self, xar_city: xr.DataArray, city_weather_df: pd.DataFrame):
        self.xar_city = xar_city
        self.city_weather_df = city_weather_df
        self.fig = go.Figure()

    def plot(self):
        xar_city_aggregated_services = self.xar_city.sum(dim=TrafficDataDimensions.SERVICE.value).sum(dim=AggregationLevel.IRIS.value)
        users_time_series = Indexing.day_time_to_datetime_index(xar=xar_city_aggregated_services).T.to_pandas()
        temperature_time_series = self.city_weather_df['temperature'].to_frame()
        precipitation_time_series = self.city_weather_df['precipitation_last_3h'].to_frame()
        self._scatter_plot(time_series=users_time_series, name="Users")
        self._scatter_plot(time_series=temperature_time_series, name="Temperature")
        self._scatter_plot(time_series=precipitation_time_series, name="Precipitation")
        self._layout()
        self.fig.show(renderer="browser")

    def _scatter_plot(self, time_series: pd.DataFrame, name: str):
        rescaled_service_time_series = MinMaxScaler().fit_transform(time_series.values.reshape(-1, 1)).flatten()
        trace_time_series = go.Scatter(x=time_series.index, y=rescaled_service_time_series, mode='lines', name=f'{name}')
        self.fig.add_trace(trace_time_series)

    def _layout(self):
        self.fig.update_layout(title_text="Users vs temperature", xaxis_rangeslider_visible=True, font=dict(size=18))
        self.fig.update_xaxes(title_text="Time")
        self.fig.update_yaxes(title_text="Rescaled value")


if __name__ == '__main__':
    SeasonPlot(xar_city=0, n_days_season=7).plot()
