from datetime import time
from typing import Iterator


from DataInterface.TrafficDataInterface import CityTrafficData
from FeatureExtraction.ServiceConsumption.ServiceConsumption import ServiceConsumption


class ServiceConsumptionAPI:
    @staticmethod
    def compute_service_consumption(traffic_data: Iterator[CityTrafficData] or CityTrafficData, start: time = time(7, 30), end: time = time(11, 59)) -> ServiceConsumption:
        if isinstance(traffic_data, CityTrafficData):
            traffic_data = [traffic_data]

        service_consumption = None
        for city_traffic_data in traffic_data:
            if service_consumption is None:
                service_consumption = ServiceConsumptionAPI.compute_service_consumption_city(traffic_data=city_traffic_data, start=start, end=end)
            else:
                service_consumption = service_consumption.join(other=ServiceConsumptionAPI.compute_service_consumption_city(traffic_data=city_traffic_data, start=start, end=end))

        return service_consumption

    @staticmethod
    def compute_service_consumption_city(traffic_data: CityTrafficData, start: time, end: time) -> ServiceConsumption:
        service_consumption_by_location_data = traffic_data.get_service_consumption_by_location(start=start, end=end)
        service_consumption = ServiceConsumption(service_consumption_data=service_consumption_by_location_data, start=start, end=end)
        return service_consumption