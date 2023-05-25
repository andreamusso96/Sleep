from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import xarray as xr

from SleepInference.SleepClassifier import SleepClassifier
from SleepInference.SleepPreprocessor import SleepPreprocessor
from Utils import City, AggregationLevel
from Logging.Loggers import SleepInferenceLogger


class SleepDetector:
    def __init__(self, xar_city: xr.DataArray, city: City, aggregation_level: AggregationLevel,  window: int):
        self.xar_city = xar_city
        self.city = city
        self.aggregation_level = aggregation_level
        self.window = window
        SleepInferenceLogger.debug(f'SleepDetector: Initialized for {city.value}')

    def calculate_sleep_tile_time_day(self) -> xr.DataArray:
        SleepInferenceLogger.debug(f'SleepDetector: Calculating sleep patterns for {self.city.value}')
        sleep_preprocessor = SleepPreprocessor(xar_city=self.xar_city, city=self.city)
        time_series = sleep_preprocessor.preprocess()
        SleepInferenceLogger.debug(f'SleepDetector: Iterating over locations for {self.city.value}')
        sleep_data_locations = Parallel(n_jobs=-1, verbose=1)(delayed(self.classify_sleep_habits_location)(time_series[location_id].to_frame(), window=self.window) for location_id in time_series.columns)
        SleepInferenceLogger.debug(f'SleepDetector: Building xarray for {self.city.value}')
        sleep_data = np.stack(sleep_data_locations, axis=0)
        coords = {self.aggregation_level.value: time_series.columns,
                  'time': time_series.index.get_level_values(1).unique(),
                  'day': time_series.index.get_level_values(0).unique()}
        dims = [self.aggregation_level.value, 'time', 'day']
        sleep_data = xr.DataArray(sleep_data, coords=coords, dims=dims)
        SleepInferenceLogger.debug(f'SleepDetector: Sleep calculation complete for {self.city.value}')
        return sleep_data

    @staticmethod
    def classify_sleep_habits_location(time_series_location: pd.DataFrame, window: int) -> np.ndarray:
        sleep_classifier = SleepClassifier(data=time_series_location, window=window)
        sleep_data = sleep_classifier.cluster()
        sleep_data.reset_index(inplace=True)
        sleep_data = sleep_data.pivot(index='time', columns='day', values='Sleep')
        return sleep_data.values
