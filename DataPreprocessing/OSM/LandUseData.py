from enum import Enum
from typing import List
from unidecode import unidecode

import osmnx as ox
import pandas as pd
import geopandas as gpd
from Utils import City


class LandUse(Enum):
    RESIDENTIAL = 'residential'
    COMMERCIAL = 'commercial'
    INDUSTRIAL = 'industrial'


class LandUseDataDownloader:
    def __init__(self, city: City, city_geometry: gpd.GeoDataFrame):
        self.city = city
        self._city_geometry = city_geometry['geometry'].iloc[0]
