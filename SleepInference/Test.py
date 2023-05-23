from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from Utils import City, AggregationLevel
from SleepInference.SleepDetector import SleepDetector


def test_data() -> Tuple[xr.DataArray, City, AggregationLevel]:
    aggregation_level = AggregationLevel.TILE
    location_ids = list(range(10))
    time = pd.timedelta_range(start='00:00:00', end='23:59:00', freq='15min')
    service = ['Facebook', 'Instagram', 'Twitter']
    days = pd.date_range(start='2019-03-16', end='2019-05-31', freq='D')
    coords = {aggregation_level.value: location_ids,
              'time': time,
              'service': service,
              'day': days,
              }
    data = np.random.rand(len(location_ids), len(time), len(service), len(days))
    xar_city = xr.DataArray(data, coords=coords, dims=[aggregation_level.value, 'time', 'service', 'day'])
    return xar_city, City.BORDEAUX, aggregation_level


def test() -> xr.DataArray:
    xar_city, city, aggregation_level = test_data()
    window = 5
    sleep_detector = SleepDetector(xar_city=xar_city, city=city, window=window, aggregation_level=aggregation_level)
    sleep_data = sleep_detector.calculate_sleep_tile_time_day()
    return sleep_data


if __name__ == '__main__':
    sleep_data = test()