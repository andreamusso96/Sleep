import os
import time
import subprocess

import mobile_traffic as mt


def submit_jobs():
    cities = [mt.City.BORDEAUX]
    for city in cities:
        submit_command = f'sbatch --mem-per-cpu=8G --ntasks=1 --cpus-per-task=8 --time=08:00:00 --wrap="python -m cluster_run {city.value}"'
        print('SUBMITTING JOB WITH COMMAND: ', submit_command)
        os.system(submit_command)
        time.sleep(2)


if __name__ == '__main__':
    submit_jobs()