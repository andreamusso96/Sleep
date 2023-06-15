import pyinstrument

from SleepDetection.Detector import Detector
from DataIO import DataIO
from Utils import City, TrafficType, AggregationLevel


class Profiler:

    @staticmethod
    def _run_program():
        xar_city = DataIO.load_traffic_data(traffic_type=TrafficType.UL_AND_DL, geo_data_type=AggregationLevel.IRIS,
                                            city=City.LYON)
        detector = Detector(xar_city=xar_city)
        detector.detect_sleep_patterns()

    @staticmethod
    def time_profile():
        time_profiler = pyinstrument.Profiler()
        time_profiler.start()
        Profiler._run_program()
        time_profiler.stop()
        time_profiler.open_in_browser()
        Profiler._save_output_time_profiler(time_profiler)

    @staticmethod
    def _save_output_time_profiler(time_profiler: pyinstrument.Profiler):
        with open('time_profiling_report.html', 'w') as f:
            f.write(time_profiler.output_html())


if __name__ == '__main__':
    Profiler.time_profile()