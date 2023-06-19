from DataInterface.DataInterface import DataInterface
from DataPreprocessing.WeatherData.Data import SunriseSunsetData, WeatherStationData
from DataInterface.GeoDataInterface import GeoMatchingAPI


class WeatherData(DataInterface):
    def __init__(self):
        super().__init__()
        self.sunrise_sunset_data = SunriseSunsetData()
        self.weather_station_data = WeatherStationData()
        self.geo_matching = GeoMatchingAPI.load_matching()

    def get_weather_station_data(self, iris: str = None, city: City = None):
        if iris is not None:
            weather_station_code = self.geo_matching.get_weather_station(iris=iris)
        elif city is not None:
            weather_station_code = self.geo_matching.get_weather_station(city=city)
        else:
            raise ValueError('Either subset or city must be provided.')
        weather_station_data = self.weather_station_data.data[
            self.weather_station_data.data['numer_sta'] == weather_station_code]
        weather_station_data = self._reformat_weather_station_data(weather_station_data=weather_station_data)
        return weather_station_data

    @staticmethod
    def _reformat_weather_station_data(weather_station_data: pd.DataFrame):
        weather_vars = {'numer_sta': 'station', 'date': 'datetime', 't': 'temperature', 'n': 'cloudiness',
                        'rr1': 'precipitation_last_1h', 'rr3': 'precipitation_last_3h',
                        'rr12': 'precipitation_last_12h', 'ww': 'weather_classification'}
        reformatted_weather_data = weather_station_data[list(weather_vars.keys())].copy()
        reformatted_weather_data.rename(columns=weather_vars, inplace=True)
        return reformatted_weather_data

    def get_sunrise_sunset_data(self, iris: str):
        city = self.geo_matching.get_city(iris=iris)
        sunrise_sunset_data = self.sunrise_sunset_data.data[self.sunrise_sunset_data.data['city'] == city]
        return sunrise_sunset_data
