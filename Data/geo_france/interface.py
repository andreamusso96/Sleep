from enum import Enum


class GeoDataType(Enum):
    TILE = 'tile'
    IRIS = 'iris'
    WEATHER_STATION = 'weather_station'
    CITY = 'city'
    POLLING_STATION = 'polling_station'


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