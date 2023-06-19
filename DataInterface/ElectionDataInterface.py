from typing import List

from DataPreprocessing.ElectionData.Data import ElectionDataComplete
from DataInterface.DataInterface import DataInterface
from DataInterface.GeoDataInterface import GeoData, GeoDataType


class ElectionData(DataInterface):
    def __init__(self):
        super().__init__()
        self._election_data_complete = ElectionDataComplete()
        self.data = self._election_data_complete.data

    def get_election_data_table(self, column: str, value: str):
        table_value_polling_station_by_column = self.data.pivot(index=GeoDataType.POLLING_STATION.value, columns=column, values=value)
        return table_value_polling_station_by_column

    def get_election_data_table_iris_by_column(self, geo_data: GeoData, subset: List[str], column: str, value: str, aggregation_method: str):
        table_value_polling_station_by_column = self.get_election_data_table(column=column, value=value)
        polling_station_iris_map = geo_data.get_geo_data(geometry=GeoDataType.IRIS, subset=subset, other_geo_data_types=GeoDataType.POLLING_STATION).set_index(GeoDataType.POLLING_STATION.value)[GeoDataType.IRIS.value]
        table_value_polling_station_by_column_with_iris_info = table_value_polling_station_by_column.merge(polling_station_iris_map, left_index=True, right_index=True, how='inner')
        table_value_iris_by_column = table_value_polling_station_by_column_with_iris_info.groupby(by=GeoDataType.IRIS.value).agg(aggregation_method)
        return table_value_iris_by_column