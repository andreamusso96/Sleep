from datetime import date, datetime, timedelta
from typing import Tuple
from enum import Enum
from joblib import Parallel, delayed

import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd

from config import DATA_PATH, N_CORES


class City(Enum):
    BORDEAUX = 'Bordeaux'
    CLERMONT_FERRAND = 'Clermont-Ferrand'
    DIJON = 'Dijon'
    FRANCE = 'France'
    GRENOBLE = 'Grenoble'
    LILLE = 'Lille'
    LYON = 'Lyon'
    MANS = 'Mans'
    MARSEILLE = 'Marseille'
    METZ = 'Metz'
    MONTPELLIER = 'Montpellier'
    NANCY = 'Nancy'
    NANTES = 'Nantes'
    NICE = 'Nice'
    ORLEANS = 'Orleans'
    PARIS = 'Paris'
    RENNES = 'Rennes'
    ROUEN = 'Rouen'
    SAINT_ETIENNE = 'Saint-Etienne'
    STRASBOURG = 'Strasbourg'
    TOULOUSE = 'Toulouse'
    TOURS = 'Tours'


# Define your dimensions class
class CityDimensions:
    city_dims = {
        'Bordeaux': (334, 342),
        'Clermont-Ferrand': (208, 268),
        'Dijon': (195, 234),
        'France': (9742, 9588),
        'Grenoble': (409, 251),
        'Lille': (330, 342),
        'Lyon': (426, 287),
        'Mans': (228, 246),
        'Marseille': (211, 210),
        'Metz': (226, 269),
        'Montpellier': (334, 327),
        'Nancy': (151, 165),
        'Nantes': (277, 425),
        'Nice': (150, 214),
        'Orleans': (282, 256),
        'Paris': (409, 346),
        'Rennes': (423, 370),
        'Rouen': (327, 373),
        'Saint-Etienne': (305, 501),
        'Strasbourg': (296, 258),
        'Toulouse': (280, 347),
        'Tours': (251, 270)
    }

    @classmethod
    def get_city_dim(cls, city):
        city_name = city.value  # Get the city name from the Enum member
        return cls.city_dims.get(city_name)


class Service(Enum):
    TWITCH = 'Twitch'
    ORANGE_TV = 'Orange_TV'
    MICROSOFT_AZURE = 'Microsoft_Azure'
    APPLE_ICLOUD = 'Apple_iCloud'
    WEB_GAMES = 'Web_Games'
    PLAYSTATION = 'PlayStation'
    TEAMVIEWER = 'TeamViewer'
    WEB_WEATHER = 'Web_Weather'
    GOOGLE_MEET = 'Google_Meet'
    TWITTER = 'Twitter'
    AMAZON_WEB_SERVICES = 'Amazon_Web_Services'
    APPLE_MUSIC = 'Apple_Music'
    APPLE_SIRI = 'Apple_Siri'
    WEB_ADS = 'Web_Ads'
    SOUNDCLOUD = 'SoundCloud'
    WIKIPEDIA = 'Wikipedia'
    MICROSOFT_SKYDRIVE = 'Microsoft_Skydrive'
    WEB_TRANSPORTATION = 'Web_Transportation'
    MICROSOFT_OFFICE = 'Microsoft_Office'
    YAHOO_MAIL = 'Yahoo_Mail'
    WEB_FOOD = 'Web_Food'
    WHATSAPP = 'WhatsApp'
    GOOGLE_MAIL = 'Google_Mail'
    YOUTUBE = 'YouTube'
    UBER = 'Uber'
    PINTEREST = 'Pinterest'
    WEB_CLOTHES = 'Web_Clothes'
    DROPBOX = 'Dropbox'
    APPLE_MAIL = 'Apple_Mail'
    WEB_ADULT = 'Web_Adult'
    DAILYMOTION = 'DailyMotion'
    INSTAGRAM = 'Instagram'
    SKYPE = 'Skype'
    CLASH_OF_CLANS = 'Clash_of_Clans'
    POKEMON_GO = 'Pokemon_GO'
    APPLE_APP_STORE = 'Apple_App_Store'
    GOOGLE_DRIVE = 'Google_Drive'
    APPLE_WEB_SERVICES = 'Apple_Web_Services'
    APPLE_ITUNES = 'Apple_iTunes'
    WEB_FINANCE = 'Web_Finance'
    FACEBOOK_LIVE = 'Facebook_Live'
    WEB_DOWNLOADS = 'Web_Downloads'
    EA_GAMES = 'EA_Games'
    WAZE = 'Waze'
    GOOGLE_DOCS = 'Google_Docs'
    APPLE_VIDEO = 'Apple_Video'
    LINKEDIN = 'LinkedIn'
    FACEBOOK_MESSENGER = 'Facebook_Messenger'
    SNAPCHAT = 'Snapchat'
    DEEZER = 'Deezer'
    NETFLIX = 'Netflix'
    FACEBOOK = 'Facebook'
    TELEGRAM = 'Telegram'
    APPLE_IMESSAGE = 'Apple_iMessage'
    MICROSOFT_STORE = 'Microsoft_Store'
    MOLOTOV = 'Molotov'
    GOOGLE_MAPS = 'Google_Maps'
    TOR = 'Tor'
    GOOGLE_PLAY_STORE = 'Google_Play_Store'
    WEB_E_COMMERCE = 'Web_e-Commerce'
    FORTNITE = 'Fortnite'
    MICROSOFT_MAIL = 'Microsoft_Mail'
    PERISCOPE = 'Periscope'
    GOOGLE_WEB_SERVICES = 'Google_Web_Services'
    SPOTIFY = 'Spotify'
    MICROSOFT_WEB_SERVICES = 'Microsoft_Web_Services'
    WEB_STREAMING = 'Web_Streaming'
    YAHOO = 'Yahoo'


class TrafficType(Enum):
    DL = 'DL'
    UL = 'UL'
    B = 'B'


class DataIO:
    @staticmethod
    def load_traffic_data(traffic_type: TrafficType, city: City, service: Service=None, day: date=None):
        if service is None and day is None:
            return DataIO._city_traffic_data(city=city, traffic_type=traffic_type)
        elif service is None and day is not None:
            return DataIO._city_day_traffic_data(city=city, day=day, traffic_type=traffic_type)
        elif service is not None and day is None:
            return DataIO._city_service_traffic_data(city=city, service=service, traffic_type=traffic_type)
        elif service is not None and day is not None:
            return DataIO._city_service_day_traffic_data(city=city, service=service, day=day, traffic_type=traffic_type)
        else:
            raise ValueError('Invalid parameters')

    @staticmethod
    def _city_traffic_data(city: City, traffic_type: TrafficType):
        days = DataIO.get_days()
        day_service_pairs = [(day, service) for day in days for service in Service]
        data_vals = Parallel(n_jobs=N_CORES)(delayed(DataIO._city_service_day_traffic_data)(city=city, service=service, day=day, traffic_type=traffic_type).values for day, service in day_service_pairs)
        data = np.stack([np.stack(data_vals[i: i + len(Service)], axis=-1) for i in range(0, len(data_vals), len(Service))], axis=-1)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_time_labels(),
                  'service': [service.value for service in Service],
                  'day': DataIO.get_day_labels()}
        dims = ['tile_id', 'time', 'service', 'day']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_service_traffic_data(city: City, service: Service, traffic_type: TrafficType):
        days = DataIO.get_days()
        data_vals = Parallel(n_jobs=N_CORES)(delayed(DataIO._city_service_day_traffic_data)(city=city, service=service, day=day, traffic_type=traffic_type).values for day in days)
        data = np.stack(data_vals, axis=-1)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_time_labels(),
                  'day': DataIO.get_day_labels()}
        dims = ['tile_id', 'time', 'day']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_day_traffic_data(city: City, day: date, traffic_type: TrafficType):
        data_vals = Parallel(n_jobs=N_CORES)(delayed(DataIO._city_service_day_traffic_data)(city=city, service=service, day=day, traffic_type=traffic_type).values for service in Service)
        data = np.stack(data_vals, axis=-1)
        coords = {'tile_id': DataIO.get_tile_ids(city=city),
                  'time': DataIO.get_time_labels(),
                  'service': [service.value for service in Service]}
        dims = ['tile_id', 'time', 'service']
        xar = xr.DataArray(data, coords=coords, dims=dims)
        return xar

    @staticmethod
    def _city_service_day_traffic_data(city: City, service: Service, day: date, traffic_type: TrafficType):
        if traffic_type == TrafficType.DL:
            return DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=False)
        elif traffic_type == TrafficType.UL:
            return DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=True)
        elif traffic_type == TrafficType.B:
            ul_data = DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=True)
            dl_data = DataIO._load_traffic_data_file(city=city, service=service, day=day, uplink=False)
            return ul_data + dl_data

    @staticmethod
    def _load_traffic_data_file(city: City, service: Service, day: date, uplink: bool):
        file_path = DataIO._get_traffic_data_file_path(city=city, service=service, day=day, uplink=uplink)
        cols = ['tile_id'] + DataIO.get_time_labels()
        traffic_data = pd.read_csv(file_path, sep=' ', names=cols)
        traffic_data.set_index('tile_id', inplace=True)
        return traffic_data

    @staticmethod
    def _get_traffic_data_file_path(city: City, service: Service, day: date, uplink: bool):
        day_str = day.strftime('%Y%m%d')
        ending = 'UL' if uplink else 'DL'
        path = f'{DATA_PATH}/{city.value}/{service.value}/{day_str}/'
        file_name = f'{city.value}_{service.value}_{day_str}_{ending}.txt'
        file_path = path + file_name
        return file_path

    @staticmethod
    def get_time_labels():
        times = [(datetime(2023, 1, 1) + timedelta(minutes=15 * i)).strftime('%H:%M') for i in range(96)]
        return times

    @staticmethod
    def get_day_labels():
        days = DataIO.get_days()
        day_labels = [day.strftime('%Y-%m-%d') for day in days]
        return day_labels

    @staticmethod
    def get_tile_ids(city: City):
        data = DataIO._load_traffic_data_file(city=city, service=Service.FACEBOOK_MESSENGER, day=date(2019, 3, 20), uplink=True)
        return data.index

    @staticmethod
    def get_days():
        return [date(2019, 3, 16) + timedelta(days=i) for i in range(77)]
