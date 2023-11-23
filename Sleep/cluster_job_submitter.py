import os
import time

import mobile_traffic as mt


def submit_jobs():
    cities = [mt.City.BORDEAUX]
    for city in cities:
        command = f'sbatch --mem=65G --cpus-per-task=8 --time=08:00:00 --wrap="python -m cluster_run {city.value}"'
        os.system(command)
        time.sleep(2)