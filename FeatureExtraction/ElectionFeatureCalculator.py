from enum import Enum
from typing import List

import pandas as pd

from FeatureExtraction.Feature import Feature
from FeatureExtraction.FeatureCalculator import FeatureCalculator


class ElectionFeatureName(Enum):
    ENTROPY = 'entropy'
    SIMPSON = 'simpson'


class ElectionFeatureCalculator(FeatureCalculator):
    def __init__(self, election_result_by_location: pd.DataFrame):
        super().__init__()
        self.result = election_result_by_location

    def get_election_feature(self, feature: ElectionFeatureName, subset: List[str] = None) -> Feature:
        result = self.result.loc[subset] if subset is not None else self.result
        if feature == ElectionFeatureName.ENTROPY:
            entropy_vals = result.apply(self.entropy, axis=1).to_frame(name=ElectionFeatureName.ENTROPY.value)
            entropy = Feature(data=entropy_vals, name=ElectionFeatureName.ENTROPY.value)
            return entropy
        elif feature == ElectionFeatureName.SIMPSON:
            simpson_vals = result.apply(self.simpson, axis=1).to_frame(name=ElectionFeatureName.SIMPSON.value)
            simpson = Feature(data=simpson_vals, name=ElectionFeatureName.SIMPSON.value)
            return simpson
        else:
            raise NotImplementedError
