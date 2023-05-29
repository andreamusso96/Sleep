from joblib import Parallel, delayed

import pandas as pd
from statsmodels.tsa.seasonal import MSTL, DecomposeResult


class Decomposer:
    def __init__(self, traffic_time_series_data: pd.DataFrame):
        self.traffic_time_series_data = traffic_time_series_data
        self.period_day = 96
        self.period_week = self.period_day*7
        self.periods = [self.period_day, self.period_week]
        self.windows = [self.period_day + 1, self.period_week + 1]

    def get_daily_component_of_traffic_time_series_data(self) -> pd.DataFrame:
        decomposed_traffic_time_series_data = Parallel(n_jobs=-1, verbose=1)(delayed(self._decompose_traffic_time_series_location)(traffic_time_series_location=self.traffic_time_series_data[col].to_frame()) for col in self.traffic_time_series_data.columns[::60])
        daily_components = [decomposed_traffic_time_series_location.seasonal[f'seasonal_{self.period_day}'].to_frame() for decomposed_traffic_time_series_location in decomposed_traffic_time_series_data]
        daily_components = pd.concat(daily_components, axis=1)
        daily_components.columns = self.traffic_time_series_data.columns[::60]
        return daily_components

    def _decompose_traffic_time_series_location(self, traffic_time_series_location: pd.DataFrame) -> DecomposeResult:
        mstl = MSTL(traffic_time_series_location, periods=self.periods, windows=self.windows, stl_kwargs={'robust': True})
        res = mstl.fit()
        return res
