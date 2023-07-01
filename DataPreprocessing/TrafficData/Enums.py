from enum import Enum
from datetime import timedelta


class TrafficType(Enum):
    DL = 'DL'
    UL = 'UL'
    UL_AND_DL = 'UL_AND_DL'
    USERS = 'Users'


class ServiceType(Enum):
    ENTERTAINMENT = 'entertainment'
    ALL = 'all'


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
    def get_services(traffic_type: TrafficType = None, service_type: ServiceType = None, return_values=False):
        assert traffic_type is not None or service_type is not None, 'Either traffic_type or service_type must be set'

        if service_type is None:
            if traffic_type == TrafficType.USERS:
                service_type = ServiceType.ENTERTAINMENT
            else:
                service_type = ServiceType.ALL

        if service_type == ServiceType.ENTERTAINMENT:
            services = [service for service in Service if Service.is_entertainment_service(service)]
        elif service_type == ServiceType.ALL:
            services = [service for service in Service]
        else:
            raise ValueError(f'Invalid service_type {service_type}')

        if return_values:
            return [service.value for service in services]
        else:
            return services

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
            data_consumption_service_in_timespan = hourly_data_consumption[service] * (timespan / timedelta(hours=1))
            return data_consumption_service_in_timespan


class TrafficDataDimensions(Enum):
    SERVICE = 'service'
    TIME = 'time'
    DAY = 'day'
    DATETIME = 'datetime'