import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Union, Callable

from mobile_data import ScreenTimeData, MobileTrafficData
import engineer_features as ef
import transform

def generate_distribution(n_samples: int, exponent: float, bound: float):
    samples = np.array([generate_sample(exponent=exponent, bound=bound) for n in range(n_samples)])
    return samples


def generate_sample(exponent: float, bound: float) -> int:
    zipf_sample = np.random.zipf(a=exponent, size=int(bound /2))
    cum_sum_sample = np.cumsum(zipf_sample)
    n = np.searchsorted(cum_sum_sample, v=bound) - 1
    return n


def random_sample(exponent: float, bound: float) -> int:
    return np.random.randint(low=0, high=bound, size=1)


if __name__ == '__main__':
    s = generate_distribution(n_samples=10000, exponent=2, bound=1000)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.log(1 + s), histnorm="probability"))
    fig.show(renderer="browser")

