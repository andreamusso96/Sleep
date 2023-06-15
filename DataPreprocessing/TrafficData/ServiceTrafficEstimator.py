from datetime import timedelta

import xarray as xr
import pandas as pd
import numpy as np
from Utils import Service, Calendar, Anomalies


class ServiceTrafficEstimator:
    def __init__(self, service_traffic_data: xr.DataArray, service: Service):
        self.service_traffic_data = service_traffic_data
        self.service = service

    def estimate_traffic_person_in_15min(self):
        traffic_data = self.remove_anomalies(total_traffic=self.service_traffic_data)
        traffic_data = self.remove_weekends(total_traffic=traffic_data)
        traffic_data = self.remove_holidays(total_traffic=traffic_data)
        traffic_samples_with_few_users = self._extract_traffic_samples_with_few_users(traffic_data=traffic_data)
        traffic_person_in_15min = 2 * np.median(traffic_samples_with_few_users)
        average_traffic_15min = np.median(self._to_time_series(traffic_data=traffic_data))
        return traffic_person_in_15min, average_traffic_15min, traffic_samples_with_few_users

    def _extract_traffic_samples_with_few_users(self, traffic_data):
        night_traffic_time_series = self.get_night_traffic(time_series_traffic_data=self._to_time_series(traffic_data=traffic_data))
        zero_traffic_mask = np.where(night_traffic_time_series == 0, 1, 0)
        traffic_samples_with_few_users_mask = np.roll(zero_traffic_mask, shift=1, axis=0) + np.roll(zero_traffic_mask, shift=-1, axis=0)
        traffic_samples_with_few_users_mask[0] = 0
        traffic_samples_with_few_users_mask[-1] = 0
        traffic_samples_with_few_users = (night_traffic_time_series * traffic_samples_with_few_users_mask).flatten()
        traffic_samples_with_few_users = traffic_samples_with_few_users[traffic_samples_with_few_users != 0]
        return traffic_samples_with_few_users

    @staticmethod
    def get_night_traffic(time_series_traffic_data):
        night_times = (time_series_traffic_data.index.get_level_values('time') >= timedelta(hours=3)) & (
                        time_series_traffic_data.index.get_level_values('time') <= timedelta(hours=5))
        night_traffic = time_series_traffic_data.loc[night_times].values
        return night_traffic


    @staticmethod
    def _to_time_series(traffic_data: xr.DataArray):
        time_series = traffic_data.stack(datetime=['day', 'time']).T.to_pandas()
        return time_series

    @staticmethod
    def remove_weekends(total_traffic: xr.DataArray) -> xr.DataArray:
        days = total_traffic.day.to_pandas()
        return total_traffic.drop_sel(day=[d for d in days if d.weekday() >= 5])

    @staticmethod
    def remove_holidays(total_traffic: xr.DataArray) -> xr.DataArray:
        return total_traffic.drop_sel(day=Calendar.holidays())

    @staticmethod
    def remove_anomalies(total_traffic: xr.DataArray) -> xr.DataArray:
        anomaly_dates = Anomalies.get_all_anomaly_dates()
        return total_traffic.drop_sel(day=anomaly_dates)


def plot_figure_15min_traffic_data(service):
    service_traffic_data = DataIO.load_traffic_data(traffic_type=TrafficType.UL_AND_DL,
                                                    geo_data_type=AggregationLevel.IRIS, service=service)
    service_traffic_estimator = ServiceTrafficEstimator(service_traffic_data=service_traffic_data, service=service)
    traffic_person_in_15min, average_traffic_15min, traffic_samples_with_few_users = service_traffic_estimator.estimate_traffic_person_in_15min()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.log(traffic_samples_with_few_users)))
    fig.update_layout(
        title_text=f'{service.value} traffic person in 15min: {np.log(traffic_person_in_15min):.2f}. \n Average traffic 15min: {np.log(average_traffic_15min):.2f} \n Average # people online {average_traffic_15min/traffic_person_in_15min:.2f}')
    fig.update_xaxes(title_text='Traffic')
    fig.update_yaxes(title_text='Frequency')
    fig.show()


if __name__ == '__main__':
    from DataIO import DataIO
    from Utils import City, Service, AggregationLevel, TrafficType
    import plotly.graph_objs as go
    services = [Service.INSTAGRAM, Service.FACEBOOK, Service.YOUTUBE, Service.WHATSAPP, Service.SKYPE, Service.TWITTER, Service.FACEBOOK_MESSENGER]
    for s in services:
        plot_figure_15min_traffic_data(service=s)



