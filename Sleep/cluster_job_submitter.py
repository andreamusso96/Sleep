import os
import time
import subprocess

import mobile_traffic as mt


def submit_jobs():
    cities = [mt.City.BORDEAUX]
    for city in cities:
        submit_command = f'sbatch --mem=65G --cpus-per-task=8 --time=08:00:00 --wrap="python -m cluster_run {city.value}"'
        print('SUBMITTING JOB WITH COMMAND: ', submit_command)
        process = subprocess.Popen(submit_command, shell=True)
        process.communicate()
        time.sleep(2)