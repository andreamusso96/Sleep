from datetime import date, datetime, timedelta
from enum import Enum

import pandas as pd

from config import DATA_PATH


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


class DataIO:
    @staticmethod
    def load_file(city: City, service: Service, day: date, uplink: bool):
        file_path = DataIO._get_file_path(city=city, service=service, day=day, uplink=uplink)
        day_dt = datetime(day.year, day.month, day.day)
        times = [day_dt + timedelta(minutes=15 * i) for i in range(96)]
        cols = ['tile_id'] + [t.strftime('%H:%M') for t in times]
        traffic_data = pd.read_csv(file_path, sep=' ', names=cols)
        traffic_data.set_index('tile_id', inplace=True)
        return traffic_data

    @staticmethod
    def _get_file_path(city: City, service: Service, day: date, uplink: bool):
        day_str = day.strftime('%Y%m%d')
        ending = 'UL' if uplink else 'DL'
        path = f'{DATA_PATH}/{city.value}/{service.value}/{day_str}/'
        file_name = f'{city.value}_{service.value}_{day_str}_{ending}.txt'
        file_path = path + file_name
        return file_path