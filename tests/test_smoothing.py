"""
Tests for the smoothing module
"""
import numpy as np
import pytest
from trialframe.smoothing import (
    norm_gauss_window,
    hw_to_std,
    beta_window,
    smooth_mat,
    only_one_is_not_None,
)


def test_norm_gauss_window_basic():
    """Test that norm_gauss_window creates a normalized window."""
    bin_length = 0.01  # 10ms bins
    std = 0.05  # 50ms standard deviation
    win = norm_gauss_window(bin_length, std, causal=False)
    
    # Window should sum to 1 (normalized)
    assert np.abs(np.sum(win) - 1.0) < 1e-10
    # Window should be non-empty
    assert len(win) > 0
    # All values should be positive
    assert np.all(win >= 0)


def test_norm_gauss_window_causal():
    """Test that causal window is one-sided."""
    bin_length = 0.01
    std = 0.05
    win_causal = norm_gauss_window(bin_length, std, causal=True)
    win_normal = norm_gauss_window(bin_length, std, causal=False)
    
    # Causal window should be shorter (one-sided)
    assert len(win_causal) < len(win_normal)
    # Should still be normalized
    assert np.abs(np.sum(win_causal) - 1.0) < 1e-10


def test_hw_to_std():
    """Test conversion from half-width to standard deviation."""
    hw = 0.1
    std = hw_to_std(hw)
    # Standard deviation should be positive and smaller than half-width
    assert std > 0
    assert std < hw
    # Verify the mathematical relationship
    expected_std = hw / (2 * np.sqrt(2 * np.log(2)))
    assert np.abs(std - expected_std) < 1e-10


def test_beta_window():
    """Test that beta_window creates a normalized window."""
    bin_length = 0.01
    std = 0.05
    alpha = 2.0
    beta_param = 5.0
    
    win = beta_window(bin_length, std, alpha=alpha, beta_param=beta_param)
    
    # Window should sum to 1 (normalized)
    assert np.abs(np.sum(win) - 1.0) < 1e-10
    # Window should be non-empty
    assert len(win) > 0
    # All values should be positive
    assert np.all(win >= 0)


def test_beta_window_different_params():
    """Test beta window with different alpha and beta parameters."""
    bin_length = 0.01
    std = 0.05
    
    win1 = beta_window(bin_length, std, alpha=2.0, beta_param=5.0)
    win2 = beta_window(bin_length, std, alpha=3.0, beta_param=5.0)
    
    # Different parameters should produce different windows
    assert not np.allclose(win1, win2)
    # Both should be normalized
    assert np.abs(np.sum(win1) - 1.0) < 1e-10
    assert np.abs(np.sum(win2) - 1.0) < 1e-10


def test_smooth_mat_1d():
    """Test smoothing a 1D array."""
    # Create a noisy signal
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
    
    dt = 0.01
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, backend='convolve1d')
    
    # Smoothed signal should have same shape
    assert smoothed.shape == signal.shape
    # Smoothed signal should have lower variance
    assert np.var(smoothed) < np.var(signal)


def test_smooth_mat_2d():
    """Test smoothing a 2D array (multiple columns)."""
    # Create a noisy 2D signal
    np.random.seed(42)
    signal = np.column_stack([
        np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1,
        np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1,
    ])
    
    dt = 0.01
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, backend='convolve1d')
    
    # Smoothed signal should have same shape
    assert smoothed.shape == signal.shape
    # Each column should have lower variance
    assert np.var(smoothed[:, 0]) < np.var(signal[:, 0])
    assert np.var(smoothed[:, 1]) < np.var(signal[:, 1])


def test_smooth_mat_causal():
    """Test causal smoothing doesn't use future data."""
    # Create a signal with a step function
    signal = np.zeros(100)
    signal[50:] = 1.0
    
    dt = 0.01
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': 0.02}, backend='convolve1d', causal=True)
    
    # Before the step, causal smoothing should not show the future step
    # (values before index 50 should be close to 0)
    assert np.all(smoothed[:48] < 0.1)
    # After the step, values should increase
    assert smoothed[52] > smoothed[48]


def test_smooth_mat_with_window():
    """Test smoothing with a custom window."""
    signal = np.random.randn(100)
    # Create a simple moving average window
    win = np.ones(5) / 5
    
    smoothed = smooth_mat(signal, win=win, backend='convolve1d')
    
    assert smoothed.shape == signal.shape


def test_smooth_mat_with_hw():
    """Test smoothing with half-width parameter."""
    signal = np.random.randn(100)
    dt = 0.01
    hw = 0.05
    
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'hw': hw}, backend='convolve1d')
    
    assert smoothed.shape == signal.shape


def test_smooth_mat_beta_kernel():
    """Test smoothing with beta kernel."""
    signal = np.random.randn(100)
    dt = 0.01
    
    smoothed = smooth_mat(
        signal,
        dt=dt,
        kernel='beta',
        kernel_params={'std': 0.05, 'alpha': 2.0, 'beta_param': 5.0},
        backend='convolve1d'
    )
    
    assert smoothed.shape == signal.shape


def test_smooth_mat_convolve_backend():
    """Test smoothing with scipy.signal.convolve backend."""
    signal = np.random.randn(100)
    dt = 0.01
    
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, backend='convolve')
    
    assert smoothed.shape == signal.shape


def test_smooth_mat_convolve_backend_2d():
    """Test smoothing 2D array with convolve backend."""
    signal = np.random.randn(100, 3)
    dt = 0.01
    
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, backend='convolve')
    
    assert smoothed.shape == signal.shape


def test_smooth_mat_invalid_backend():
    """Test that invalid backend raises error."""
    signal = np.random.randn(100)
    dt = 0.01
    
    with pytest.raises(AssertionError):
        smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, backend='invalid')


def test_smooth_mat_invalid_dimensions():
    """Test that 3D array raises error."""
    signal = np.random.randn(10, 10, 10)
    dt = 0.01
    
    with pytest.raises(ValueError, match="mat has to be a 1D or 2D array"):
        smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, backend='convolve1d')


def test_smooth_mat_invalid_kernel():
    """Test that invalid kernel raises error."""
    signal = np.random.randn(100)
    dt = 0.01
    
    with pytest.raises(ValueError, match="kernel must be 'gaussian' or 'beta'"):
        smooth_mat(signal, dt=dt, kernel_params={'std': 0.05}, kernel='invalid')


def test_only_one_is_not_None():
    """Test the helper function only_one_is_not_None."""
    assert only_one_is_not_None([None, 1, None]) is True
    assert only_one_is_not_None([None, None, None]) is False
    assert only_one_is_not_None([1, 2, None]) is False
    assert only_one_is_not_None([1, 2, 3]) is False


def test_smooth_mat_no_dt_with_window():
    """Test that dt is not required when window is provided."""
    signal = np.random.randn(100)
    win = np.ones(5) / 5
    
    # Should work without dt when window is provided
    smoothed = smooth_mat(signal, win=win, backend='convolve1d')
    assert smoothed.shape == signal.shape


def test_smooth_mat_no_dt_without_window():
    """Test that dt is required when window is not provided."""
    signal = np.random.randn(100)
    
    with pytest.raises(AssertionError):
        smooth_mat(signal, kernel_params={'std': 0.05}, backend='convolve1d')
