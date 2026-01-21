import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

class SoftnormScaler(TransformerMixin, BaseEstimator):
    def __init__(self, norm_const=5):
        self.norm_const = norm_const

    def fit(self,X,y=None):
        def get_range(arr,axis=None):
            return np.nanmax(arr,axis=axis)-np.nanmin(arr,axis=axis)
        self.activity_range_ = get_range(X,axis=0)
        return self

    def transform(self,X):
        check_is_fitted(self, 'activity_range_')
        return X / (self.activity_range_ + self.norm_const)

class DataFrameTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self,X,y=None):
        self.transformer.fit(X,y)
        return self

    def transform(self, X):
        output = self.transformer.transform(X)
        return pd.DataFrame(
            output,
            index=X.index,
            columns=range(output.shape[1]),
        )
