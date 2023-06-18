from datetime import date, timedelta
from typing import List
from enum import Enum

import xarray as xr
import pandas as pd
import numpy as np


class City(Enum):
    BORDEAUX = 'Bordeaux'
    CLERMONT_FERRAND = 'Clermont-Ferrand'
    DIJON = 'Dijon'
    GRENOBLE = 'Grenoble'
    LILLE = 'Lille'
    LYON = 'Lyon'
    MANS = 'Mans'  #
    MARSEILLE = 'Marseille'
    METZ = 'Metz'
    MONTPELLIER = 'Montpellier'
    NANCY = 'Nancy'
    NANTES = 'Nantes'
    NICE = 'Nice'
    ORLEANS = 'Orleans'  #
    PARIS = 'Paris'
    RENNES = 'Rennes'
    SAINT_ETIENNE = 'Saint-Etienne'  #
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


class TrafficType(Enum):
    DL = 'DL'
    UL = 'UL'
    UL_AND_DL = 'UL_AND_DL'
    USERS = 'Users'


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

    @staticmethod
    def get_services(traffic_type: TrafficType, return_values=False):
        if traffic_type == TrafficType.USERS:
            if return_values:
                return [service.value for service in Service if Service.is_entertainment_service(service)]
            else:
                return [service for service in Service if Service.is_entertainment_service(service)]
        else:
            if return_values:
                return [service.value for service in Service]
            else:
                return [service for service in Service]

    @staticmethod
    def is_entertainment_service(service):
        _entertainment_services = {
            Service.TWITCH,
            Service.ORANGE_TV,
            Service.WEB_GAMES,
            Service.WEB_WEATHER,
            Service.TWITTER,
            Service.APPLE_MUSIC,
            Service.WEB_ADS,
            Service.SOUNDCLOUD,
            Service.WIKIPEDIA,
            Service.WEB_FOOD,
            Service.YOUTUBE,
            Service.PINTEREST,
            Service.WEB_CLOTHES,
            Service.WEB_ADULT,
            Service.DAILYMOTION,
            Service.INSTAGRAM,
            Service.CLASH_OF_CLANS,
            Service.POKEMON_GO,
            Service.WEB_FINANCE,
            Service.FACEBOOK_LIVE,
            Service.EA_GAMES,
            Service.APPLE_VIDEO,
            Service.LINKEDIN,
            Service.SNAPCHAT,
            Service.DEEZER,
            Service.NETFLIX,
            Service.FACEBOOK,
            Service.MOLOTOV,
            Service.WEB_E_COMMERCE,
            Service.FORTNITE,
            Service.PERISCOPE,
            Service.SPOTIFY,
            Service.WEB_STREAMING,
            Service.YAHOO
        }
        return service in _entertainment_services

    @staticmethod
    def get_service_data_consumption(service, timespan: timedelta = timedelta(minutes=15)):
        if Service.is_entertainment_service(service):
            hourly_data_consumption = {
                Service.TWITCH: 800,
                Service.ORANGE_TV: 900,
                Service.WEB_GAMES: 100,
                Service.WEB_WEATHER: 5,
                Service.TWITTER: 100,
                Service.APPLE_MUSIC: 150,
                Service.WEB_ADS: 30,
                Service.SOUNDCLOUD: 150,
                Service.WIKIPEDIA: 15,
                Service.WEB_FOOD: 50,
                Service.YOUTUBE: 800,
                Service.PINTEREST: 200,
                Service.WEB_CLOTHES: 60,
                Service.WEB_ADULT: 800,
                Service.DAILYMOTION: 800,
                Service.INSTAGRAM: 200,
                Service.CLASH_OF_CLANS: 100,
                Service.POKEMON_GO: 60,
                Service.WEB_FINANCE: 50,
                Service.FACEBOOK_LIVE: 800,
                Service.EA_GAMES: 100,
                Service.APPLE_VIDEO: 900,
                Service.LINKEDIN: 100,
                Service.SNAPCHAT: 200,
                Service.DEEZER: 150,
                Service.NETFLIX: 900,
                Service.FACEBOOK: 200,
                Service.MOLOTOV: 900,
                Service.WEB_E_COMMERCE: 60,
                Service.FORTNITE: 100,
                Service.PERISCOPE: 800,
                Service.SPOTIFY: 150,
                Service.WEB_STREAMING: 800,
                Service.YAHOO: 100
            }
            data_consumption_service_in_timespan = hourly_data_consumption[service] * (timespan/timedelta(hours=1))
            return data_consumption_service_in_timespan


class AggregationLevel(Enum):
    TILE = 'tile'
    IRIS = 'iris'


class TrafficDataDimensions(Enum):
    SERVICE = 'service'
    TIME = 'time'
    DAY = 'day'
    DATETIME = 'datetime'


class Calendar:
    @staticmethod
    def holidays() -> List[date]:
        holidays = [date(2019, 4, 19), date(2019, 4, 22), date(2019, 5, 1), date(2019, 5, 8), date(2019, 5, 30)]
        return holidays

    @staticmethod
    def fridays_and_saturdays() -> List[date]:
        days = pd.date_range(start='2019-03-16', end='2019-05-31')
        weekends = [day.date() for day in days if day.dayofweek in [4, 5]]
        return weekends


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

    @staticmethod
    def get_all_anomaly_dates():
        anomaly_dates = [date(2019, 4, 9), date(2019, 4, 12), date(2019, 4, 14), date(2019, 5, 12),
                         date(2019, 5, 22), date(2019, 5, 23), date(2019, 5, 24), date(2019, 5, 25)]
        return anomaly_dates


class Indexing:
    @staticmethod
    def day_time_to_datetime_index(xar: xr.DataArray) -> xr.DataArray:
        new_index = np.add.outer(xar.indexes[TrafficDataDimensions.DAY.value], xar.indexes[TrafficDataDimensions.TIME.value]).flatten()
        datetime_xar = xar.stack(datetime=[TrafficDataDimensions.DAY.value, TrafficDataDimensions.TIME.value], create_index=False)
        datetime_xar = datetime_xar.reindex({'datetime': new_index})
        return datetime_xar


if __name__ == '__main__':
    Calendar.fridays_and_saturdays()