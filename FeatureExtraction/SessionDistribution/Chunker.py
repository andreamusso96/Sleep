from datetime import time, datetime, timedelta

import pandas as pd
import numpy as np


class Chunker:
    def __init__(self, chunk_start_time: time, chunk_end_time: time):
        self.start_of_chunk = chunk_start_time
        self.end_of_chunk = chunk_end_time
        if self.start_of_chunk > self.end_of_chunk:
            self.day_difference = timedelta(days=1)
        else:
            self.day_difference = timedelta(days=0)

    def chunk_series(self, traffic_time_series):
        dates = self._get_dates(traffic_time_series=traffic_time_series)
        chunks = [traffic_time_series.loc[datetime.combine(date=date, time=self.start_of_chunk): datetime.combine(date=date + self.day_difference, time=self.end_of_chunk)] for date in dates]
        chunks = [chunk for chunk in chunks if not chunk.empty]
        return chunks

    def _get_dates(self, traffic_time_series: pd.DataFrame):
        chunk_times_mask = (traffic_time_series.index.time >= self.start_of_chunk) | (traffic_time_series.index.time <= self.end_of_chunk)
        chunk_dates = np.unique(traffic_time_series.index.date[chunk_times_mask])
        return chunk_dates