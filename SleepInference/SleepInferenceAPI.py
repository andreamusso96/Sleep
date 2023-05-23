from typing import Generator
import xarray as xr

from SleepInference.SleepDetector import SleepDetector
from DataIO import DataIO
from Utils import City, TrafficType
from Logging.Loggers import SleepInferenceLogger


class SleepInferenceAPI:
    @staticmethod
    def detect_sleep_patterns_city(city: City, window: int) -> xr.DataArray:
        SleepInferenceLogger.info(f'SleepInferenceAPI: Detecting sleep patterns for {city.value}')
        SleepInferenceLogger.debug(f'SleepInferenceAPI: Loading traffic data for {city.value}')
        xar_city = DataIO.load_traffic_data(city=city, traffic_type=TrafficType.B)
        SleepInferenceLogger.debug(f'SleepInferenceAPI: Calculating sleep patterns for {city.value}')
        sleep_detector = SleepDetector(xar_city=xar_city, city=city, window=window)
        sleep_tile_time_day = sleep_detector.calculate_sleep_tile_time_day()
        SleepInferenceLogger.info(f'SleepInferenceAPI: Sleep patterns detected for {city.value}')
        return sleep_tile_time_day

    @staticmethod
    def detect_sleep_patterns(window: int):
        for city in City:
            yield SleepInferenceAPI.detect_sleep_patterns_city(city=city, window=window)
