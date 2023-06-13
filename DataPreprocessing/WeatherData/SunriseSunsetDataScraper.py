from typing import List
import itertools

import requests
from bs4 import BeautifulSoup
import pandas as pd


from Utils import City


class SunriseSunsetScraper:
    def __init__(self, cities: List[City], months: List[str]):
        self.cities = cities
        self.months = months

    def scrape(self) -> pd.DataFrame:
        city_month_combinations = list(itertools.product(self.cities, self.months))
        data = [self.scrape_data_city_month(city=city, month=month) for city, month in city_month_combinations]
        data = pd.concat(data, axis=0, ignore_index=True)
        return data

    def scrape_data_city_month(self, city: City, month: str) -> pd.DataFrame:
        raw_data = self.fetch_raw_data_city_month(city=city, month=month)
        data = self.parse_data_city_month(content=raw_data, city=city)
        return data

    @staticmethod
    def fetch_raw_data_city_month(city: City, month: str) -> str or None:
        url = f"http://calendriersolaire.com/calendrier?location={city.value}&month={month}&year={2019}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None

    @staticmethod
    def parse_data_city_month(content, city: City) -> pd.DataFrame:
        soup = BeautifulSoup(content, 'html.parser')
        data = []
        for td in soup.find_all("td"):
            # Skip if it doesn't have 'day' class
            if "day" not in td.get("class", []):
                continue

            # Extract date, sunrise and sunset
            day = td.get("rel", "")
            sunrise = td.find("span", class_="sunrise").text.split(' ')[0]
            sunset = td.find("span", class_="sunset").text.split(' ')[0]

            # Convert to datetime objects
            day = pd.to_datetime(day, format="%d-%m-%Y").date()
            sunrise = pd.to_datetime(sunrise, format="%H:%M").time()
            sunset = (pd.to_datetime(sunset, format="%H:%M") + pd.DateOffset(hours=12)).time()

            # Add to data
            data.append([city.value, day, sunrise, sunset])

        data = pd.DataFrame(data=data, columns=["city", "day", "sunrise", "sunset"])
        return data


def scrape():
    cities = [city for city in City]
    months = ['Mars', 'Avril', 'Mai']
    scraper = SunriseSunsetScraper(cities=cities, months=months)
    data = scraper.scrape()
    return data


