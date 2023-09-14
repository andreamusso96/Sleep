from datetime import date
from typing import List

import pandas as pd
import xarray as xr
import numpy as np

from .enums import City, TrafficDataDimensions


class Calendar:
    @staticmethod
    def holidays() -> List[date]:
        holidays = [date(2019, 4, 19), date(2019, 4, 22), date(2019, 5, 1), date(2019, 5, 8), date(2019, 5, 30)]
        return holidays

    @staticmethod
    def fridays_and_saturdays() -> List[date]:
        days = pd.date_range(start='2019-03-16', end='2019-05-31')
        weekends = [day.date() for day in days if day.dayofweek in [4, 5]]
        return weekends


class Anomalies:
    @staticmethod
    def get_anomaly_dates_by_city(city: City):
        if city == City.BORDEAUX:
            anomaly_dates = [date(2019, 4, 9), date(2019, 4, 12), date(2019, 4, 14), date(2019, 5, 12),
                             date(2019, 5, 22), date(2019, 5, 23), date(2019, 5, 24), date(2019, 5, 25)]
        elif city == City.TOULOUSE:
            anomaly_dates = [date(2019, 4, 12), date(2019, 4, 14), date(2019, 5, 12), date(2019, 5, 22),
                             date(2019, 5, 23), date(2019, 5, 24), date(2019, 5, 25)]
        else:
            anomaly_dates = [date(2019, 4, 14), date(2019, 5, 12)]
        return anomaly_dates

    @staticmethod
    def get_all_anomaly_dates():
        anomaly_dates = [date(2019, 4, 9), date(2019, 4, 12), date(2019, 4, 14), date(2019, 5, 12),
                         date(2019, 5, 22), date(2019, 5, 23), date(2019, 5, 24), date(2019, 5, 25)]
        return anomaly_dates


def day_time_to_datetime_index(xar: xr.DataArray) -> xr.DataArray:
    new_index = np.add.outer(xar.indexes[TrafficDataDimensions.DAY.value], xar.indexes[TrafficDataDimensions.TIME.value]).flatten()
    datetime_xar = xar.stack(datetime=[TrafficDataDimensions.DAY.value, TrafficDataDimensions.TIME.value], create_index=False)
    datetime_xar = datetime_xar.reindex({TrafficDataDimensions.DATETIME.value: new_index})
    return datetime_xar
