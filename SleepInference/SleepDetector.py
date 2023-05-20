import numpy as np
from joblib import Parallel, delayed

import pandas as pd
import xarray as xr

from SleepInference.SleepClassifier import SleepClassifier
from SleepInference.SleepPreprocessor import SleepPreprocessor
from Utils import City
from Logging.Loggers import SleepInferenceLogger


class SleepDetector:
    def __init__(self, xar_city: xr.DataArray, city: City, window: int):
        self.xar_city = xar_city
        self.city = city
        self.window = window
        SleepInferenceLogger.debug(f'SleepDetector: Initialized for {city.value}')

    def calculate_sleep_tile_time_day(self) -> xr.DataArray:
        SleepInferenceLogger.debug(f'SleepDetector: Calculating sleep patterns for {self.city.value}')
        sleep_preprocessor = SleepPreprocessor(xar_city=self.xar_city, city=self.city)
        time_series = sleep_preprocessor.preprocess()
        SleepInferenceLogger.debug(f'SleepDetector: Iterating over tiles for {self.city.value}')
        sleep_data_tiles = Parallel(n_jobs=-1, verbose=1)(delayed(self.classify_sleep_habits_tile)(time_series[tile_id].to_frame(), window=self.window) for tile_id in time_series.columns)
        SleepInferenceLogger.debug(f'SleepDetector: Building xarray for {self.city.value}')
        sleep_data = np.stack(sleep_data_tiles, axis=0)
        coords = {'tile_id': time_series.columns,
                  'time': time_series.index.get_level_values(1).unique(),
                  'day': time_series.index.get_level_values(0).unique()}
        dims = ['tile_id', 'time', 'day']
        sleep_data = xr.DataArray(sleep_data, coords=coords, dims=dims)
        SleepInferenceLogger.debug(f'SleepDetector: Sleep calculation complete for {self.city.value}')
        return sleep_data

    @staticmethod
    def classify_sleep_habits_tile(time_series_tile: pd.DataFrame, window: int) -> np.ndarray:
        sleep_classifier = SleepClassifier(data=time_series_tile, window=window)
        sleep_data = sleep_classifier.cluster()
        sleep_data.reset_index(inplace=True)
        sleep_data = sleep_data.pivot(index='time', columns='day', values='Sleep')
        return sleep_data.values