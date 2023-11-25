import os
import time
import subprocess

import mobile_traffic as mt


def submit_jobs():
    args = ['rca', 'nsi', 'other']
    mem_per_cpu = ['8G', '8G', '4G']
    for i in range(len(args)):
        submit_command = f'sbatch --mem-per-cpu={mem_per_cpu[i]} --ntasks=1 --cpus-per-task=8 --time=14:00:00 --wrap="python -m cluster_run {args[i]}"'
        print('SUBMITTING JOB WITH COMMAND: ', submit_command)
        os.system(submit_command)
        time.sleep(2)


if __name__ == '__main__':
    submit_jobs()