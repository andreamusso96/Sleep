from typing import Generator
import xarray as xr

from SleepInference.SleepDetector import SleepDetector
from DataIO import DataIO
from Utils import City, TrafficType


class SleepInferenceAPI:
    @staticmethod
    def detect_sleep_patterns_city(city: City, window: int) -> xr.DataArray:
        xar_city = DataIO.load_traffic_data(city=city, traffic_type=TrafficType.B)
        sleep_detector = SleepDetector(xar_city=xar_city, city=city, window=window)
        sleep_tile_time_day = sleep_detector.calculate_sleep_tile_time_day()
        return sleep_tile_time_day

    @staticmethod
    def detect_sleep_patterns(window: int) -> Generator[xr.DataArray]:
        for city in City:
            yield SleepInferenceAPI.detect_sleep_patterns_city(city=city, window=window)
