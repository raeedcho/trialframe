import numpy as np

import scipy.signal as scs
from scipy.ndimage import convolve1d
from scipy.stats import beta as beta_dist

def norm_gauss_window(bin_length, std, causal=False):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_length : float
        binning length of the array we want to smooth in s
    std : float
        standard deviation of the window
        use hw_to_std to calculate std based from half-width
    causal : bool (default False)
        If True, create a one-sided Gaussian (causal) kernel
        that only includes past data

    Returns
    -------
    win : 1D np.array
        Gaussian kernel with mass normalized to 1
    """
    if causal:
        # One-sided Gaussian: only uses current and past data
        n_samples = int(10*std/bin_length)
        full_win = scs.windows.gaussian(n_samples, std/bin_length)
        # Take the right half to make it one-sided (weighted towards past)
        win = full_win[n_samples//2:]
    else:
        win = scs.windows.gaussian(int(10*std/bin_length), std/bin_length)
    return win / np.sum(win)


def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))


def beta_window(bin_length, std, alpha=2.0, beta_param=5.0):
    """
    Beta distribution window for causal smoothing.
    
    The beta distribution creates a one-sided kernel that decays into the past,
    making it suitable for causal filtering.

    Parameters
    ----------
    bin_length : float
        binning length of the array we want to smooth in s
    std : float
        standard deviation-like parameter controlling window length
    alpha : float (default 2.0)
        Alpha parameter of beta distribution (controls shape)
        Lower values create sharper kernels, higher values create smoother decay
    beta_param : float (default 5.0)
        Beta parameter of beta distribution (controls how much weight is in the tail)
        Higher values put more weight towards the current time

    Returns
    -------
    win : 1D np.array
        Beta-distribution kernel with mass normalized to 1
    """
    n_samples = int(10*std/bin_length)
    # Create a one-sided beta distribution (from 0 to 1)
    x = np.linspace(0, 1, n_samples)
    win = beta_dist.pdf(x, alpha, beta_param)
    return win / np.sum(win)


def smooth_mat(mat, dt=None, win=None, backend='convolve1d', causal=False, kernel='gaussian', kernel_params=None):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    dt : float
        length of the timesteps in seconds
    win : 1D array-like (optional)
        smoothing window to convolve with (if provided, other kernel parameters are ignored)
    backend: str, either 'convolve1d' or 'convolve' (default 'convolve1d')
        'convolve1d' uses scipy.ndimage.convolve1d, which is faster in some cases
        'convolve'  uses scipy.signal.convolve, which may scale better for large arrays
    causal : bool (default False)
        If True, use causal (one-sided) kernel to avoid acausal smoothing effects
    kernel : str (default 'gaussian')
        Type of kernel: 'gaussian' or 'beta'
    kernel_params : dict (optional)
        Parameters for the kernel function. Can include:
        - 'std': float - standard deviation of the smoothing window (default 0.05)
        - 'hw': float - half-width of the smoothing window (alternative to std)
        - 'alpha': float - for beta kernel (default 2.0)
        - 'beta_param': float - for beta kernel (default 5.0)

    Returns
    -------
    np.array of the same size as mat
    """
    if kernel_params is None:
        kernel_params = {}
    
    # Extract std/hw from kernel_params
    std = kernel_params.get('std')
    hw = kernel_params.get('hw')
    
    assert only_one_is_not_None((win, hw, std))
    assert backend=='convolve' or backend=='convolve1d', 'backend must be either convolve or convolve1d'

    if win is None:
        assert dt is not None, "specify dt if not supplying window"

        if hw is None:
            if std is None:
                std = 0.05
        else:
            assert (std is None), "only give hw or std"
            std = hw_to_std(hw)

        # Build params for kernel function (excluding std/hw)
        kernel_kws = {k: v for k, v in kernel_params.items() if k not in ['std', 'hw']}
        
        if kernel == 'gaussian':
            win = norm_gauss_window(dt, std, causal=causal)
        elif kernel == 'beta':
            win = beta_window(dt, std, **kernel_kws)
        else:
            raise ValueError("kernel must be 'gaussian' or 'beta'")

    if mat.ndim != 1 and mat.ndim != 2:
        raise ValueError("mat has to be a 1D or 2D array")
    
    # For causal kernels, use constant mode to avoid future data leakage
    # and set origin to align the end of the kernel with the current point
    mode = 'constant' if causal else 'reflect'
    cval = 0 if causal else None
    origin = -(len(win) - 1) // 2 if causal else 0
        
    if backend == 'convolve1d':
        if causal:
            return convolve1d(mat, win, axis=0, output=np.float32, mode=mode, cval=cval, origin=origin)
        else:
            return convolve1d(mat, win, axis=0, output=np.float32, mode=mode)
    elif backend == 'convolve':
        if mat.ndim == 1:
            return scs.convolve(mat, win, mode='same')
        elif mat.ndim == 2:
            return np.column_stack([scs.convolve(mat[:,i], win, mode='same') for i in range(mat.shape[1])])
    else:
        raise ValueError("backend has to either 'convolve1d' or 'convolve'")

def only_one_is_not_None(args):
    return sum([arg is not None for arg in args]) == 1