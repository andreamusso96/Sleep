import pandas as pd

from . import config

data = None
metadata = None


def load_data():
    global data
    global metadata
    load_metadata()
    if data is None:
        data = pd.read_csv(config.get_data_file_path(), index_col=0)


def load_metadata():
    global metadata
    if metadata is None:
        metadata = pd.read_csv(config.get_metadata_file_path())