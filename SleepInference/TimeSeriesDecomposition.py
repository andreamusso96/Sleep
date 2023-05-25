from joblib import Parallel, delayed
from typing import Dict, List

import pandas as pd
from statsmodels.tsa.seasonal import MSTL, DecomposeResult

from config import TEMP_DATA_PATH

class TimeSeriesDecomposer:
    def __init__(self, time_series: pd.DataFrame, periods: List[int], windows: List[int]):
        self.time_series = time_series
        self.periods = periods
        self.windows = windows

    def decompose(self, return_df=False) -> DecomposeResult or pd.DataFrame:
        mstl = MSTL(self.time_series, periods=self.periods, windows=self.windows, stl_kwargs={'robust': True})
        res = mstl.fit()
        if return_df:
            return self._build_dataframe(result=res)
        else:
            return res

    def _build_dataframe(self, result: DecomposeResult) -> pd.DataFrame:
        seasonal_day = result.seasonal[f'seasonal_{self.periods[0]}']
        seasonal_week = result.seasonal[f'seasonal_{self.periods[1]}']
        decomposition_summary = pd.concat([seasonal_day, seasonal_week, result.observed, result.trend, result.resid], axis=1)
        decomposition_summary.columns = ['seasonal_day', 'seasonal_week', 'observed', 'trend', 'residuals']
        return decomposition_summary



class Decomposer:
    def __init__(self, time_series_data: pd.DataFrame):
        self.time_series_data = time_series_data
        self._ts_data = self._transform_index_to_datetime_index(time_series=self.time_series_data)
        self.periods = [96, 96*7]
        self.windows = [96 + 1, 96*7 + 1]

    def decompose(self) -> Dict[str, pd.DataFrame]:
        decomposed_time_series_data = Parallel(n_jobs=-1, verbose=1)(delayed(self._decompose_location_time_series)(time_series=self._ts_data[col].to_frame()) for col in self._ts_data.columns)
        decomposed_time_series_dict = {col: decomposed_time_series_data[i] for i, col in enumerate(self._ts_data.columns)}
        return decomposed_time_series_dict

    def _decompose_location_time_series(self, time_series: pd.DataFrame) -> pd.DataFrame:
        ts_decomposer = TimeSeriesDecomposer(time_series=time_series, periods=self.periods, windows=self.windows)
        decomposed_time_series = ts_decomposer.decompose(return_df=True)
        decomposed_time_series = self._reset_index_to_original_index(time_series=decomposed_time_series)
        # TODO: REMOVE THE LINE BELOW
        decomposed_time_series.to_csv(f'{TEMP_DATA_PATH}/SeasonalDec/{time_series.columns[0]}.csv', index=True)
        return decomposed_time_series

    @staticmethod
    def _transform_index_to_datetime_index(time_series: pd.DataFrame) -> pd.DataFrame:
        datetime_index = time_series.index.get_level_values(0) + time_series.index.get_level_values(1)
        time_series_with_datetime_index = pd.DataFrame(data=time_series.values, index=datetime_index, columns=time_series.columns)
        return time_series_with_datetime_index

    def _reset_index_to_original_index(self, time_series: pd.DataFrame) -> pd.DataFrame:
        time_series.index = self.time_series_data.index
        return time_series

