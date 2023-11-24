import os
import time
import subprocess

import mobile_traffic as mt


def submit_jobs():
    args = ['rca', 'nsi', 'other']
    for arg in args:
        submit_command = f'sbatch --mem-per-cpu=2G --ntasks=1 --cpus-per-task=8 --time=14:00:00 --wrap="python -m cluster_run {arg}"'
        print('SUBMITTING JOB WITH COMMAND: ', submit_command)
        os.system(submit_command)
        time.sleep(2)


if __name__ == '__main__':
    submit_jobs()