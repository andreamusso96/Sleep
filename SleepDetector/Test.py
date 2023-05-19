import numpy as np
import pandas as pd
import xarray as xr

from Utils import City
from SleepDetector import SleepDetector


def test_data():
    tile_ids = list(range(10))
    time = pd.timedelta_range(start='00:00:00', end='23:59:00', freq='15min')
    service = ['Facebook', 'Instagram', 'Twitter']
    days = pd.date_range(start='2019-03-16', end='2019-05-31', freq='D')
    coords = {'tile_id': tile_ids,
              'time': time,
              'service': service,
              'day': days,
              }
    data = np.random.rand(len(tile_ids), len(time), len(service), len(days))
    xar_city = xr.DataArray(data, coords=coords, dims=['tile_id', 'time', 'service', 'day'])
    return xar_city, City.BORDEAUX


def test():
    xar_city, city = test_data()
    window = 5
    sleep_detector = SleepDetector(xar_city=xar_city, city=city, window=window)
    sleep_data = sleep_detector.calculate_sleep_tile_time_day()
    return sleep_data


if __name__ == '__main__':
    sleep_data = test()