from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []
def fit(self, X, y=None):
    return self
def transform(self, X):
 X2 = X.copy()
 return X2.drop(columns=self.cols_to_drop, errors='ignore')