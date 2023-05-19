import pandas as pd
import xarray as xr

from Utils import City, Anomalies, Calendar


class SleepPreprocessor:
    def __init__(self, xar_city: xr.DataArray, city: City):
        self.xar_city = xar_city
        self.city = city

    def preprocess(self) -> pd.DataFrame:
        total_traffic = self.total_traffic()
        total_traffic = self.remove_anomalies(total_traffic=total_traffic, city=self.city)
        total_traffic = self.remove_weekends(total_traffic=total_traffic)
        total_traffic = self.remove_holidays(total_traffic=total_traffic)
        time_series = self.to_time_series_format(total_traffic=total_traffic)
        return time_series

    def total_traffic(self) -> xr.DataArray:
        return self.xar_city.sum(dim='service')

    @staticmethod
    def remove_weekends(total_traffic: xr.DataArray) -> xr.DataArray:
        days = total_traffic.day.to_pandas()
        return total_traffic.drop_sel(day=[d for d in days if d.weekday() >= 5])

    @staticmethod
    def remove_holidays(total_traffic: xr.DataArray) -> xr.DataArray:
        return total_traffic.drop_sel(day=Calendar.holidays())

    @staticmethod
    def remove_anomalies(total_traffic: xr.DataArray, city: City) -> xr.DataArray:
        anomaly_dates = Anomalies.get_anomaly_dates_by_city(city=city)
        return total_traffic.drop_sel(day=anomaly_dates)

    @staticmethod
    def to_time_series_format(total_traffic: xr.DataArray) -> pd.DataFrame:
        time_series = total_traffic.stack(datetime=['day', 'time']).T.to_pandas()
        return time_series