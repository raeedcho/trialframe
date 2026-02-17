# trialframe

A python package for manipulating trial-based time series data in neuroscience experiments.

## Overview

`trialframe` provides a structured framework for working with trial-based experimental data, particularly suited for neuroscience and behavioral experiments. The package uses pandas DataFrames with multi-level indices to organize data by trials, time, and experimental states, making it easy to slice, transform, and analyze time series data across experimental conditions.

## Key Features

- **Time-based slicing**: Extract data based on time windows and experimental events
- **State management**: Track and manipulate experimental states and transitions
- **Signal processing**: Smooth data, estimate derivatives, and compute kinematic features
- **Trial alignment**: Reindex trials relative to specific events or time points
- **Hierarchical data manipulation**: Work with multi-level indexed DataFrames for complex experimental designs
- **Dimensionality reduction**: Built-in support for demixed PCA (dPCA) analysis
- **Scikit-learn integration**: Custom transformers for preprocessing pipelines

## Installation

### From source

```bash
git clone <repository-url>
cd trialframe
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

This installs additional dependencies for testing and development (pytest, black, isort).

## Requirements

- Python ≥ 3.10
- numpy
- pandas
- scipy
- scikit-learn
- xarray
- dpca

## Core Concepts

### Trialframe Format

A trialframe is a pandas DataFrame with a multi-level index typically containing:
- `trial_id`: Unique identifier for each trial
- `time`: Timestamp (usually as pd.Timedelta)
- `state` (optional): Experimental state or phase

Columns can be single-level or multi-level (e.g., `('signal', 'channel')`) for hierarchical organization of different signals and their channels.

## Modules

### `timeseries`

Functions for time series analysis and kinematic processing:

- `get_sample_spacing(df)`: Determine the sampling rate from time index
- `smooth_data(data, std)`: Apply Gaussian smoothing to data
- `estimate_kinematic_derivative(trial_signal, deriv, cutoff)`: Compute derivatives using Butterworth filtering
- `estimate_kinematic_derivative_savgol(trial_signal, deriv, window_length, polyorder)`: Compute derivatives using Savitzky-Golay filter
- `remove_baseline(trial, ref_event, ref_slice)`: Subtract baseline activity
- `hold_mahal_distance(points, reference)`: Compute Mahalanobis distance from reference distribution

### `time_slice`

Functions for slicing and reindexing data by time:

- `slice_by_time(data, time_slice, timecol)`: Extract data within a time window
- `state_list_to_transitions(state_list, timecol)`: Convert state list to transition times
- `state_transitions_to_list(state_transitions, new_index)`: Convert transitions back to full state list
- `reindex_trial_from_time(trial, reference_time, timecol)`: Shift time index relative to a reference time
- `reindex_trial_from_event(trial, event, timecol)`: Align trials to a specific event
- `get_epoch_data(trial, epochs)`: Extract data for specific experimental epochs

### `munge`

Data manipulation utilities for hierarchical DataFrames:

- `get_index_level(df, level)`: Extract specific index levels as series
- `multivalue_xs(df, keys, level, axis)`: Cross-section with multiple keys
- `hierarchical_assign(df, assign_dict)`: Extend pandas assign() for multi-level columns

### `state_space`

Preprocessing and transformation tools:

- `SoftnormScaler`: Scikit-learn transformer for soft normalization
- `DataFrameTransformer`: Wrapper to preserve DataFrame structure through sklearn transformers

### `smoothing`

Low-level smoothing functions:

- `norm_gauss_window(bin_length, std)`: Create normalized Gaussian kernel
- `hw_to_std(hw)`: Convert half-width to standard deviation
- `smooth_mat(mat, dt, std, hw, win, backend)`: Smooth 1D or 2D arrays

### `dpca`

Demixed Principal Component Analysis utilities:

- `make_dpca_tensor(trialframe, conditions, channel_level_name)`: Convert trialframe to dPCA tensor format
- `DPCAWrapper`: Scikit-learn compatible wrapper for dPCA

## Usage Examples

### Basic time slicing

```python
import pandas as pd
import trialframe as tf

# Slice data from 0 to 500ms
time_window = slice(pd.to_timedelta('0ms'), pd.to_timedelta('500ms'))
windowed_data = tf.slice_by_time(trial_data, time_window)
```

### Align trials to an event

```python
# Reindex all trials so time=0 at 'Go Cue' event
aligned_trials = tf.reindex_trial_from_event(trial_data, event='Go Cue')
```

### Smooth and compute velocity

```python
# Smooth position data
smooth_pos = tf.smooth_data(position_data, std=pd.to_timedelta('50ms'))

# Estimate velocity
velocity = tf.estimate_kinematic_derivative(smooth_pos, deriv=1, cutoff=30)
```

### Hierarchical column assignment

```python
# Add computed signals to DataFrame with hierarchical columns
enhanced_data = tf.hierarchical_assign(
    trial_data,
    {
        'velocity': lambda df: compute_velocity(df['position']),
        'acceleration': lambda df: compute_acceleration(df['velocity'])
    }
)
```

### Extract specific trials by condition

```python
# Get only trials from specific tasks
subset = tf.multivalue_xs(trial_data, keys=['CST', 'RTT'], level='task')
```

## Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black trialframe/
isort trialframe/
```

## Author

Raeed Chowdhury

## License

See LICENSE file for details.

---

*This README was generated by GitHub Copilot.*