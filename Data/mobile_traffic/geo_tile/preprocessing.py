import geopandas as gpd

from . import config
from ..enums import City


def load_tile_geo_data():
    tile_geo_data = {}
    for city in City:
        tile_geo_data_city = load_tile_geo_data_city(city=city)
        tile_geo_data[city] = tile_geo_data_city
    return tile_geo_data


def load_tile_geo_data_city(city: City):
    file_path = config.get_data_file_path(city=city)
    data = gpd.read_file(filename=file_path, dtypes={'tile_id': int})
    data.rename(columns={'tile_id': 'tile'}, inplace=True)
    data.set_index(keys='tile', inplace=True)
    return data