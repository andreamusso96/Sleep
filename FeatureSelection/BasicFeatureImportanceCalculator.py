import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler


class BasicFeatureImportanceCalculator:
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.labels = labels

    def compute_feature_importance(self) -> pd.DataFrame:
        feature_importance = self._estimate_feature_importance(features=self.features, labels=self.labels)
        return feature_importance

    @staticmethod
    def _estimate_feature_importance(features: pd.DataFrame, labels: pd.DataFrame):
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