import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
from typing import Optional
from .state_space import DataFrameTransformer
from sklearn.utils.validation import check_is_fitted
from dPCA.dPCA import dPCA

def make_dpca_tensor(trialframe: pd.DataFrame,conditions: list[str],channel_level_name: str = 'channel') -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    """
    xr_obj = (
        trialframe
        .groupby(conditions)
        .mean()
        .stack()
        .reorder_levels([channel_level_name] + conditions)
        .to_xarray()
    )
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.to_numpy()
    # If it's a Dataset (e.g., due to top-level column names), collapse to DataArray
    return xr_obj.to_array().squeeze().to_numpy()

def make_dpca_tensor_simple(trialframe: pd.DataFrame,conditions: list[str],channel_level_name: str = 'channel') -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    Note: This version for some reason does not output the same tensor as make_dpca_tensor.
    dPCA also works worse on it for some reason.
    """
    xr_obj = (
        trialframe
        .stack()
        .groupby([channel_level_name] + conditions).mean()
        .to_xarray()
    )
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.to_numpy()
    return xr_obj.to_array().squeeze().to_numpy()

def make_dpca_trial_tensor(trialframe: pd.DataFrame,conditions: list[str],channel_level_name: str = 'channel') -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    This version outputs a tensor with trials as the first dimension,
    for dPCA auto-regularization.

    It produces bad results for some reason...
    """
    tensor = (
        trialframe
        .stack()
        .to_frame(name='activity') # type: ignore
        .assign(**{
            'trial num': lambda df: df.groupby([channel_level_name]+conditions).cumcount(),
        })
        .set_index('trial num',append=True)
        .groupby(['trial num',channel_level_name] + conditions).mean()
        .to_xarray()
        ['activity']
        .to_numpy()
    )

    return tensor

class PhaseConcatDPCA(DataFrameTransformer):
    """
    dPCA model for trialframe data that concatenates phases of trials along the time dimension.
    Note: I originally wrote this because I thought dPCA would be better than using PCA
    on marginalized data to find CIS and direction-specific components, but it turns out
    marginalized PCA works better, for some inexplicable reason.
    """
    def __init__(self,conditions: list[str], protect: list[str]=['t'], channel_level_name: str = 'channel', **dpca_kwargs):
        assert conditions[-1] == 'time', "Last condition must be 'time' for PhaseConcatDPCA"
        assert 'labels' in dpca_kwargs, "Must provide 'labels' in dpca_kwargs"
        assert dpca_kwargs['labels'][-1] == 't', "Last label must be 't' for PhaseConcatDPCA"

        self.conditions = conditions
        self.protect = protect
        self.channel_level_name = channel_level_name
        transformer = dPCA(**dpca_kwargs)
        transformer.protect = self.protect  # Protect the time condition
        super().__init__(transformer=transformer)

    def fit(self, X, y=None):
        tensor = (
            X
            .groupby('phase')
            .apply(
                lambda df: make_dpca_tensor(df,conditions=self.conditions, channel_level_name=self.channel_level_name)
            )
            .pipe(np.concatenate,axis=-1)  # Concatenate along the time dimension
        )
        self.transformer.fit(X=tensor)
        self.is_fitted_ = True  # Mark as fitted
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, 'transformer')

        out = self.transformer.transform(X.values.T)
        return pd.concat(
            {
                marg: pd.DataFrame(
                    marg_proj.T,
                    index=X.index,
                )
                for marg,marg_proj in out.items()
            },
            axis=1,
        )