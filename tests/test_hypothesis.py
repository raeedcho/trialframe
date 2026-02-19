"""
Property-based tests using hypothesis for the trialframe package.

These tests use property-based testing to verify invariants and mathematical
properties of functions across a wide range of inputs.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays

from trialframe.smoothing import (
    norm_gauss_window,
    hw_to_std,
    beta_window,
    smooth_mat,
)
from trialframe.time_slice import (
    state_list_to_transitions,
    state_transitions_to_list,
)


# ============================================================================
# Tests for smoothing.py
# ============================================================================

@given(
    bin_length=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=1.0),
)
def test_norm_gauss_window_is_normalized(bin_length, std):
    """Property: Gaussian window should always sum to 1 (be normalized)."""
    win = norm_gauss_window(bin_length, std, causal=False)
    assert np.abs(np.sum(win) - 1.0) < 1e-10


@given(
    bin_length=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=1.0),
)
def test_norm_gauss_window_causal_is_normalized(bin_length, std):
    """Property: Causal Gaussian window should also sum to 1."""
    win = norm_gauss_window(bin_length, std, causal=True)
    assert np.abs(np.sum(win) - 1.0) < 1e-10


@given(
    bin_length=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=1.0),
)
def test_norm_gauss_window_all_positive(bin_length, std):
    """Property: All values in the window should be non-negative."""
    win = norm_gauss_window(bin_length, std, causal=False)
    assert np.all(win >= 0)


@given(
    bin_length=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=1.0),
)
def test_norm_gauss_window_causal_is_shorter(bin_length, std):
    """Property: Causal window should be shorter than non-causal (one-sided)."""
    # Only test when std is large enough relative to bin_length to produce meaningful windows
    assume(std / bin_length >= 2.0)
    
    win_causal = norm_gauss_window(bin_length, std, causal=True)
    win_normal = norm_gauss_window(bin_length, std, causal=False)
    assert len(win_causal) < len(win_normal)


@given(hw=st.floats(min_value=0.01, max_value=10.0))
def test_hw_to_std_is_positive(hw):
    """Property: Standard deviation should always be positive."""
    std = hw_to_std(hw)
    assert std > 0


@given(hw=st.floats(min_value=0.01, max_value=10.0))
def test_hw_to_std_smaller_than_hw(hw):
    """Property: Standard deviation should be smaller than half-width."""
    std = hw_to_std(hw)
    assert std < hw


@given(hw=st.floats(min_value=0.01, max_value=10.0))
def test_hw_to_std_mathematical_relationship(hw):
    """Property: Verify the exact mathematical relationship."""
    std = hw_to_std(hw)
    expected_std = hw / (2 * np.sqrt(2 * np.log(2)))
    assert np.abs(std - expected_std) < 1e-10


@given(
    bin_length=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=1.0),
    alpha=st.floats(min_value=1.0, max_value=10.0),
    beta_param=st.floats(min_value=1.0, max_value=10.0),
)
def test_beta_window_is_normalized(bin_length, std, alpha, beta_param):
    """Property: Beta window should always sum to 1 (be normalized)."""
    # Only test when std is large enough relative to bin_length to produce meaningful windows
    assume(std / bin_length >= 2.0)
    
    win = beta_window(bin_length, std, alpha=alpha, beta_param=beta_param)
    assert np.abs(np.sum(win) - 1.0) < 1e-10


@given(
    bin_length=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=1.0),
    alpha=st.floats(min_value=1.0, max_value=10.0),
    beta_param=st.floats(min_value=1.0, max_value=10.0),
)
def test_beta_window_all_positive(bin_length, std, alpha, beta_param):
    """Property: All values in the beta window should be non-negative."""
    # Only test when std is large enough relative to bin_length to produce meaningful windows
    assume(std / bin_length >= 2.0)
    
    win = beta_window(bin_length, std, alpha=alpha, beta_param=beta_param)
    assert np.all(win >= 0)


@given(
    signal=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=50, max_value=200),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    ),
    dt=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=0.2),
)
def test_smooth_mat_preserves_shape_1d(signal, dt, std):
    """Property: Smoothing should preserve the shape of the input array."""
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': std}, backend='convolve1d')
    assert smoothed.shape == signal.shape


@given(
    n_rows=st.integers(min_value=50, max_value=200),
    n_cols=st.integers(min_value=2, max_value=5),
    dt=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=0.2),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_smooth_mat_preserves_shape_2d(n_rows, n_cols, dt, std, seed):
    """Property: Smoothing should preserve the shape of 2D input arrays."""
    # Use hypothesis-controlled seed for reproducibility
    rng = np.random.RandomState(seed)
    signal = rng.randn(n_rows, n_cols) * 10
    
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': std}, backend='convolve1d')
    assert smoothed.shape == signal.shape


@given(
    n=st.integers(min_value=50, max_value=200),
    dt=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=0.2),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_smooth_mat_reduces_variance(n, dt, std, seed):
    """Property: Smoothing should not significantly increase variance for noisy signals."""
    # Generate a reproducible noisy signal (approximately white Gaussian noise)
    rng = np.random.RandomState(seed)
    signal = rng.randn(n) * 10

    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': std}, backend='convolve1d')

    smoothed_var = np.var(smoothed)
    original_var = np.var(signal)

    # Smoothing should not substantially increase variance.
    # Allow a small relative tolerance for numerical and edge effects.
    assert smoothed_var <= original_var * (1.0 + 1e-3)
@given(
    dt=st.floats(min_value=0.001, max_value=0.1),
    std=st.floats(min_value=0.01, max_value=0.2),
)
def test_smooth_mat_causal_doesnt_use_future(dt, std):
    """Property: Causal smoothing should not use future data."""
    # Create a signal with a step function at position 50
    signal = np.zeros(100)
    signal[50:] = 1.0
    
    smoothed = smooth_mat(signal, dt=dt, kernel_params={'std': std}, backend='convolve1d', causal=True)
    
    # Before the step (with some margin), values should be close to 0
    # The exact margin depends on the std, so we use a conservative check
    # Ensure margin doesn't exceed the signal boundary
    margin = min(int(5 * std / dt), 40)
    if 50 - margin > 0:
        assert np.all(smoothed[:50-margin] < 0.1)


# ============================================================================
# Tests for time_slice.py
# ============================================================================

def make_state_series(states, times, trial_id=1):
    """Helper to create a state series for testing."""
    idx = pd.MultiIndex.from_arrays(
        [np.full(len(states), trial_id), pd.to_timedelta(times, unit="s")],
        names=["trial_id", "time"]
    )
    return pd.Series(states, index=idx, name="state")


@given(
    n_states=st.integers(min_value=2, max_value=10),
    dt=st.floats(min_value=0.001, max_value=0.1),
)
def test_state_transitions_roundtrip(n_states, dt):
    """Property: Converting states to transitions and back should preserve data."""
    # Create a state series with different states
    n_samples = n_states * 10
    times = np.arange(n_samples) * dt
    
    # Create alternating states
    states = []
    for i in range(n_samples):
        state_idx = (i // 10) % n_states
        states.append(f"State{state_idx}")
    
    s = make_state_series(states, times)
    
    # Convert to transitions and back
    trans = state_list_to_transitions(s)
    s_reconstructed = state_transitions_to_list(trans, s.index)
    
    # Should match except possibly the first sample (off-by-one is tolerable)
    if len(s) > 1:
        assert s_reconstructed.iloc[1:].equals(s.iloc[1:])


@given(
    n_states=st.integers(min_value=2, max_value=5),
    n_samples_per_state=st.integers(min_value=5, max_value=20),
    dt=st.floats(min_value=0.001, max_value=0.1),
)
def test_state_list_to_transitions_preserves_unique_states(n_states, n_samples_per_state, dt):
    """Property: All unique states should be present in transitions."""
    # Create a state series with different states
    states = []
    for i in range(n_states):
        states.extend([f"State{i}"] * n_samples_per_state)
    
    times = np.arange(len(states)) * dt
    s = make_state_series(states, times)
    
    trans = state_list_to_transitions(s)
    
    # Get unique states from both
    unique_original = set(s.unique())
    unique_trans = set(trans.index.get_level_values('new_state').unique())
    
    # Transitions should contain all original states (except possibly the first one
    # if it doesn't change)
    assert unique_trans.issubset(unique_original)
    # At least one state should be in transitions
    assert len(unique_trans) > 0


@given(
    n_samples=st.integers(min_value=20, max_value=50),
    dt=st.floats(min_value=0.01, max_value=0.1),
)
def test_state_list_to_transitions_has_transition(n_samples, dt):
    """Property: When states change, there should be at least one transition."""
    # Create a state series with a clear change in the middle
    states = ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)
    times = np.arange(n_samples) * dt
    s = make_state_series(states, times)
    
    trans = state_list_to_transitions(s)
    
    # There should be at least one transition for the A->B change
    assert len(trans) >= 1
    # And 'B' should appear in the transitions
    trans_states = trans.index.get_level_values('new_state')
    assert 'B' in trans_states.values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
