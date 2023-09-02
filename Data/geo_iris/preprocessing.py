import geopandas as gpd

from . import config


def load_iris_geo_data() -> gpd.GeoDataFrame:
    data = gpd.read_file(filename=config.get_data_file_path(), dtypes={'CODE_IRIS': str})
    data.to_crs(crs='WGS 84', inplace=True)
    data = data[['CODE_IRIS', 'geometry']].copy()
    data.rename(columns={'CODE_IRIS': 'iris'}, inplace=True)
    data.set_index('iris', inplace=True)
    return data