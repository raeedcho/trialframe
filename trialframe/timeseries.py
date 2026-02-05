import pandas as pd
import numpy as np
import scipy.signal as scs
from . import munge
from .time_slice import slice_by_time, reindex_trial_from_event
from .smoothing import smooth_mat

def get_sample_spacing(df: pd.DataFrame) -> float:
    time_diffs = munge.get_index_level(df, 'time').diff()
    most_common_diff = time_diffs.value_counts().idxmax()
    
    # Ensure most_common_diff is a timedelta
    if isinstance(most_common_diff, pd.Timedelta):
        sample_spacing = most_common_diff.total_seconds()
    else:
        raise ValueError("The most common time difference is not a timedelta object")
    
    return sample_spacing

def smooth_data(data: pd.DataFrame, std: pd.Timedelta = pd.to_timedelta('50ms')) -> pd.DataFrame:
    """
    Smooth each column of the input DataFrame using a Gaussian filter.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to be smoothed.
    std : pd.Timedelta
        The standard deviation of the Gaussian kernel as a pandas Timedelta.

    Returns
    -------
    pd.DataFrame
        The smoothed data.
    """
    
    # check monotonicity of time index (single trial)
    time_index = munge.get_index_level(data, 'time')
    if not time_index.is_monotonic_increasing:
        raise ValueError("'time' index must be monotonically increasing to smooth a single trial")

    sample_spacing = get_sample_spacing(data)
    return pd.DataFrame(
        smooth_mat(
            data.values,
            dt=sample_spacing,
            std=std.total_seconds(),
            backend='convolve1d',
        ),
        index=data.index,
        columns=data.columns,
    )

def estimate_kinematic_derivative(
        trial_signal: pd.DataFrame,
        deriv: int = 1,
        cutoff: float = 30
    ) -> pd.DataFrame:
    # assert that columns are not a multiindex, so we know we're working with one "signal"
    assert not isinstance(trial_signal.columns, pd.MultiIndex), 'must work with only one "signal"'
    assert deriv == 1, 'only first derivative is supported currently'

    sample_spacing = get_sample_spacing(trial_signal)
    samprate = 1/sample_spacing
    nyquist = samprate / 2
    normalized_cutoff = cutoff / nyquist
    assert 0 < normalized_cutoff < 1, 'cutoff frequency must be between 0 and Nyquist frequency'
    filt_b, filt_a = scs.butter(4, normalized_cutoff, 'low',output='ba') # type: ignore
    return (
        trial_signal
        .pipe(lambda x: pd.DataFrame(
            scs.filtfilt(filt_b, filt_a, x.values, axis=0),
            columns=x.columns,
            index=x.index
        ))
        .pipe(lambda x: pd.DataFrame(
            np.gradient(x.values, sample_spacing, axis=0),
            columns=x.columns,
            index=x.index
        ))
    )

def estimate_kinematic_derivative_savgol(
        trial_signal: pd.DataFrame,
        deriv: int = 1,
        window_length: int = 31,
        polyorder: int = 7,
    ) -> pd.DataFrame:
    """ Estimate the kinematic derivative using Savitzky-Golay filter.

    """
    # assert that columns are not a multiindex, so we know we're working with one "signal"
    assert not isinstance(trial_signal.columns, pd.MultiIndex), 'must work with only one "signal"'

    return (
        trial_signal
        .pipe(lambda x: pd.DataFrame(
            scs.savgol_filter(
                x.values,
                window_length,
                polyorder,
                deriv=deriv,
                delta=get_sample_spacing(x),
                axis=0,
            ),
            columns=x.columns,
            index=x.index
        ))
    )

def remove_baseline(
    tf: pd.DataFrame,
    ref_event: str,
    ref_slice: slice,
    timecol: str = 'time',
) -> pd.DataFrame:
    """
    Remove the baseline (found from time slice w.r.t. an event) from the trial frame.

    Parameters
    ----------
    tf : pd.DataFrame
        The trial frame containing the data.
    ref_event : str
        The event to use as a reference for baseline removal.
    ref_slice : slice
        The time slice to use for baseline removal.
    timecol : str, optional
        The name of the time column, by default 'time'.

    Returns
    -------
    pd.DataFrame
        The trial frame with the baseline removed.
    """
    baseline = (
        tf
        .groupby('trial_id',group_keys=False)
        .apply(reindex_trial_from_event,event=ref_event,timecol=timecol) # type: ignore
        .pipe(slice_by_time,time_slice=ref_slice,timecol=timecol)
        .groupby('trial_id')
        .agg(lambda s: np.nanmean(s,axis=0))
    )
    return tf - baseline
