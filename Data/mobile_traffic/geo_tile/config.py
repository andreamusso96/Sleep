from ..enums import City

data_folder = '/Users/anmusso/Desktop/PhD/NetMob/NetMobData/data/GeoData/TileGeo/'


def get_data_file_path(city: City) -> str:
    return f'{data_folder}/{city.value}.geojson'