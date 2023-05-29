import numpy as np
import pandas as pd

from SleepDetection.Preprocessing import Chunker


class SleepScorer:
    def __init__(self, sleep_change_points: pd.DataFrame, traffic_time_series_data: pd.DataFrame):
        self.sleep_change_points = sleep_change_points
        self.traffic_time_series_data = traffic_time_series_data

    def compute_sleep_score(self):
        chunker = Chunker()
        traffic_data_chunked_by_nights = chunker.chunk_series_into_nights(traffic_time_series_data=self.traffic_time_series_data)
        sleep_score_night = [self._compute_sleep_score_night(traffic_data_night=traffic_data_night) for traffic_data_night in traffic_data_chunked_by_nights]
        sleep_score_night = pd.concat(sleep_score_night, axis=0)
        return sleep_score_night

    def _compute_sleep_score_night(self, traffic_data_night) -> pd.DataFrame:
        date = traffic_data_night.index[0].date()
        sleep_change_points_night = self.sleep_change_points.loc[date]
        sleep_score_night = [self._compute_sleep_score_night_location(sleep_change_points_night_location=sleep_change_points_night[location_id].to_frame(), traffic_data_night_location=traffic_data_night[location_id].to_frame()) for location_id in traffic_data_night.columns]
        sleep_score_night = pd.concat(sleep_score_night, axis=1)
        return sleep_score_night

    def _compute_sleep_score_night_location(self, sleep_change_points_night_location: pd.DataFrame, traffic_data_night_location: pd.DataFrame) -> pd.DataFrame:
        time_lowest_night_traffic = traffic_data_night_location.idxmin(axis=0).values[0]
        time_asleep = sleep_change_points_night_location.loc['asleep'].values[0]
        time_awake = sleep_change_points_night_location.loc['awake'].values[0]
        sleep_score_falling_asleep = self._compute_sleep_score_night_location_falling_asleep(time_asleep=time_asleep, time_lowest_night_traffic=time_lowest_night_traffic, traffic_data_night_location=traffic_data_night_location)
        sleep_score_waking_up = self._compute_sleep_score_night_location_waking_up(time_lowest_night_traffic=time_lowest_night_traffic, time_awake=time_awake, traffic_data_night_location=traffic_data_night_location)
        sleep_score_night_location = self._build_sleep_score_df(sleep_score_falling_asleep=sleep_score_falling_asleep, sleep_score_waking_up=sleep_score_waking_up, location_id=traffic_data_night_location.columns[0], time_index=traffic_data_night_location.index)
        return sleep_score_night_location

    @staticmethod
    def _build_sleep_score_df(sleep_score_falling_asleep, sleep_score_waking_up, location_id, time_index):
        sleep_score_night_location = pd.DataFrame(data=np.zeros(shape=(len(time_index), 1)),
                                                  index=time_index,
                                                  columns=[location_id])
        sleep_score_night_location.loc[sleep_score_falling_asleep.index, location_id] = sleep_score_falling_asleep.values.flatten()
        sleep_score_night_location.loc[sleep_score_waking_up.index, location_id] = sleep_score_waking_up.values.flatten()
        return sleep_score_night_location

    def _compute_sleep_score_night_location_falling_asleep(self, time_asleep, time_lowest_night_traffic, traffic_data_night_location):
        traffic_falling_asleep = traffic_data_night_location.loc[time_asleep:time_lowest_night_traffic]
        sleep_score_falling_asleep = self._compute_fraction_of_the_population_sleeping(traffic_data_values=traffic_falling_asleep.values.flatten(), threshold_everyone_awake=traffic_falling_asleep.values[0], threshold_everyone_asleep=traffic_falling_asleep.values[-1])
        sleep_score_falling_asleep = np.maximum.accumulate(sleep_score_falling_asleep)
        sleep_score_falling_asleep = pd.DataFrame(data=sleep_score_falling_asleep, index=traffic_falling_asleep.index, columns=traffic_falling_asleep.columns)
        return sleep_score_falling_asleep

    def _compute_sleep_score_night_location_waking_up(self, time_lowest_night_traffic, time_awake, traffic_data_night_location):
        traffic_waking_up = traffic_data_night_location.loc[time_lowest_night_traffic:time_awake]
        sleep_score_waking_up = self._compute_fraction_of_the_population_sleeping(traffic_data_values=traffic_waking_up.values.flatten(), threshold_everyone_awake=traffic_waking_up.values[-1], threshold_everyone_asleep=traffic_waking_up.values[0])
        sleep_score_waking_up = np.minimum.accumulate(sleep_score_waking_up)
        sleep_score_waking_up = pd.DataFrame(data=sleep_score_waking_up, index=traffic_waking_up.index, columns=traffic_waking_up.columns)
        return sleep_score_waking_up

    @staticmethod
    def _compute_fraction_of_the_population_sleeping(traffic_data_values: np.ndarray, threshold_everyone_awake, threshold_everyone_asleep) -> pd.DataFrame:
        fraction_of_population_awake = (traffic_data_values - threshold_everyone_asleep) / (threshold_everyone_awake - threshold_everyone_asleep)
        fraction_of_population_awake = np.clip(fraction_of_population_awake, a_min=0, a_max=1)
        fraction_of_population_asleep = 1 - fraction_of_population_awake
        return fraction_of_population_asleep


if __name__ == '__main__':
    from datetime import datetime
    sleep_change_points = pd.read_csv(filepath_or_buffer='Temp/change_points.csv')
    tuples = [(datetime.fromisoformat(d).date(), i)for d, i in zip(sleep_change_points['date'], sleep_change_points['sleep_state'])]
    sleep_change_points.index = pd.MultiIndex.from_tuples(tuples, names=['date', 'sleep_state'])
    sleep_change_points.drop(columns=['date', 'sleep_state'], inplace=True)
    traffic_time_series_data = pd.read_csv(filepath_or_buffer='Temp/daily_components.csv', index_col=0, parse_dates=True)
    sleep_scorer = SleepScorer(sleep_change_points=sleep_change_points, traffic_time_series_data=traffic_time_series_data)
    sleep_score = sleep_scorer.compute_sleep_score()


