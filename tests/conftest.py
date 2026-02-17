"""
Fixtures for trialframe tests
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def longer_timeseries_df():
    """
    Create a longer timeseries DataFrame for testing.
    Contains a trial with 101 time points at 10ms intervals (1 second total).
    """
    n = 101
    trial_ids = [1] * n
    times = pd.to_timedelta(np.arange(n) * 0.01, unit="s")
    idx = pd.MultiIndex.from_arrays([trial_ids, times], names=["trial_id", "time"])
    # Linear ramp from 0 to 1
    df = pd.DataFrame({"x": np.linspace(0, 1, n)}, index=idx)
    return df


@pytest.fixture
def small_state_series():
    """
    Create a small state series for testing state transitions.
    """
    n = 20
    trial_ids = [1] * n
    times = pd.to_timedelta(np.arange(n) * 0.01, unit="s")
    idx = pd.MultiIndex.from_arrays([trial_ids, times], names=["trial_id", "time"])
    # Change state at halfway point
    states = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    return pd.Series(states, index=idx, name="state")


@pytest.fixture
def simple_points_and_reference():
    """
    Create simple points and reference data for Mahalanobis distance testing.
    """
    # Create some random points
    np.random.seed(42)
    pts = pd.DataFrame(
        np.random.randn(10, 3),
        columns=["x", "y", "z"]
    )
    # Create reference distribution
    ref = pd.DataFrame(
        np.random.randn(20, 3) * 0.5,
        columns=["x", "y", "z"]
    )
    return pts, ref
