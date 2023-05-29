import os

is_cluster = False
if 'cluster' in os.path.abspath(__file__):
    is_cluster = True

if is_cluster:
    DATA_PATH = '/cluster/work/gess/coss/users/anmusso/NetMob'
else:
    DATA_PATH = '/Users/anmusso/Desktop/PhD/NetMob/data'

GEO_DATA_PATH = f'{DATA_PATH}/GeoData'
TRAFFIC_DATA_PATH = f'{DATA_PATH}/TrafficData'
TEMP_DATA_PATH = f'{DATA_PATH}/TempData'
ADMIN_DATA_PATH = f'{DATA_PATH}/AdministrativeData'