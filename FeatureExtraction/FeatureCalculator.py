from typing import Union
import numpy as np
import pandas as pd


class FeatureCalculator:
    def __init__(self):
        pass

    @staticmethod
    def entropy(probs: Union[pd.DataFrame, pd.Series]):
        vals = probs.values
        return -1 * np.dot(vals, np.log(vals, out=np.zeros_like(vals), where=vals != 0))

    @staticmethod
    def simpson(probs: Union[pd.DataFrame, pd.Series]):
        vals = probs.values
        return 1 - np.dot(vals, vals)