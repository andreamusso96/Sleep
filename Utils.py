from datetime import date
from typing import List
from enum import Enum


class City(Enum):
    BORDEAUX = 'Bordeaux'
    CLERMONT_FERRAND = 'Clermont-Ferrand'
    DIJON = 'Dijon'
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


class AggregationLevel(Enum):
    TILE = 'tile'
    IRIS = 'iris'


class Calendar:
    @staticmethod
    def holidays() -> List[date]:
        holidays = [date(2019, 4, 19), date(2019, 4, 22), date(2019, 5, 1), date(2019, 5, 8), date(2019, 5, 30)]
        return holidays


class Anomalies:
    @staticmethod
    def get_anomaly_dates_by_city(city: City):
        if city == City.BORDEAUX:
            anomaly_dates = [date(2019, 4, 9), date(2019, 4, 12), date(2019, 4, 14), date(2019, 5, 12),
                             date(2019, 5, 22), date(2019, 5, 23), date(2019, 5, 24), date(2019, 5, 25)]
        elif city == City.TOULOUSE:
            anomaly_dates = [date(2019, 4, 12), date(2019, 4, 14), date(2019, 5, 12), date(2019, 5, 22),
                             date(2019, 5, 23), date(2019, 5, 24), date(2019, 5, 25)]
        else:
            anomaly_dates = [date(2019, 4, 14), date(2019, 5, 12)]
        return anomaly_dates
