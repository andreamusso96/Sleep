from . enums import City

import geopandas as gpd
import pandas as pd
from unidecode import unidecode

def get_tile_geo_data(city: City) -> gpd.GeoDataFrame:
