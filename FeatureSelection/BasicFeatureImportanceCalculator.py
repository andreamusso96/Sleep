from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from ExpectedBedTime.ExpectedBedTimeAPI import ExpectedBedTime
from DataPreprocessing.AdminData.AdminDataComplete import AdminData
from config import ADMIN_DATA_PATH


class BasicFeatureImportanceCalculator:
    def __init__(self, expected_bed_time: ExpectedBedTime, admin_data: AdminData):
        self.expected_bed_time = expected_bed_time
        self.admin_data = admin_data

    def compute_feature_importance(self, n_quantiles: int = 5):
        features = self._get_features(iris_codes=self.expected_bed_time.expected_bed_times.index)
        labels = self.expected_bed_time.assign_iris_to_quantile(n_quantiles=n_quantiles).loc[features.index]
        feature_importance = self._estimate_feature_importance(features=features, labels=labels)
        feature_importance = self._add_feature_description(feature_importance=feature_importance)
        return feature_importance

    def _add_feature_description(self, feature_importance: pd.DataFrame):
        feature_importance['description'] = [self.admin_data.get_variable_description(var_name=var_name) for var_name in feature_importance.index]
        return feature_importance

    def _get_features(self, iris_codes: List[str]):
        features = self.admin_data.get_admin_data(iris_codes=iris_codes)
        features.dropna(axis=0, inplace=True, how='any')
        return features

    @staticmethod
    def _estimate_feature_importance(features, labels):
        f_statistic, p_values = f_classif(X=features, y=labels.values.flatten())
        mutual_information = mutual_info_classif(X=features, y=labels.values.flatten(), random_state=0)
        regression = BasicFeatureImportanceCalculator.run_regression(features=features, labels=labels)
        feature_importance = pd.DataFrame({'f_statistic': f_statistic, 'p_values': p_values, 'mutual_information': mutual_information, 'regression': regression}, index=features.columns)
        return feature_importance

    @staticmethod
    def run_regression(features, labels):
        scaled_features = StandardScaler().fit_transform(features)
        coeff = [LinearRegression().fit(X=scaled_features[:,i].reshape(-1,1), y=labels.values.flatten()).coef_[0] for i in range(scaled_features.shape[1])]
        return coeff