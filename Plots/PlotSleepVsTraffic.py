import plotly.graph_objs as go
from Utils import City

class PlotSleepVsTraffic:
    def __init__(self, xar_city_traffic, xar_city_sleep, city):
        self.xar_city_traffic = xar_city_traffic
        self.xar_city_sleep = xar_city_sleep
        self.city = city