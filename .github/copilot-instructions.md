# Copilot Instructions for trialframe

## Repository Overview

`trialframe` is a Python package for manipulating trial-based time series data in neuroscience experiments. The package uses pandas DataFrames with multi-level indices to organize data by trials, time, and experimental states.

## Project Structure

```
trialframe/
├── trialframe/           # Main package source code
│   ├── __init__.py       # Package initialization and public API
│   ├── timeseries.py     # Time series analysis and kinematic processing
│   ├── time_slice.py     # Time-based slicing and reindexing
│   ├── munge.py          # Data manipulation utilities for hierarchical DataFrames
│   ├── state_space.py    # Preprocessing and transformation tools
│   ├── smoothing.py      # Low-level smoothing functions
│   └── dpca.py           # Demixed PCA analysis utilities
├── tests/                # Test suite using pytest
│   ├── conftest.py       # Shared test fixtures
│   └── test_*.py         # Test modules mirroring source structure
├── docs/                 # Documentation
├── pyproject.toml        # Project configuration and dependencies
└── setup.py              # Setup script
```

## Development Guidelines

### Code Style

- **Formatting**: Use `black` for code formatting and `isort` for import sorting
- **Python Version**: Minimum Python 3.10
- **Type Hints**: Use type hints where it improves code clarity

### Testing

- **Framework**: pytest
- **Location**: All tests are in the `tests/` directory
- **Naming**: Test files follow the pattern `test_*.py` and mirror the source module structure
- **Fixtures**: Shared fixtures are defined in `tests/conftest.py`
- **Run Tests**: Use `pytest tests/` to run the test suite

### Dependencies

Core dependencies:
- numpy: Array operations
- pandas: DataFrame and time series handling
- scipy: Scientific computing and signal processing
- scikit-learn: Machine learning transformers and pipelines
- xarray: Multi-dimensional labeled arrays
- dpca: Demixed Principal Component Analysis

Development dependencies (install with `pip install -e ".[dev]"`):
- pytest: Testing framework
- black: Code formatter
- isort: Import sorter

## Core Concepts

### Trialframe Format

A trialframe is a pandas DataFrame with a multi-level index:
- `trial_id`: Unique identifier for each trial
- `time`: Timestamp (usually as pd.Timedelta)
- `state` (optional): Experimental state or phase

Columns can be single-level or multi-level (e.g., `('signal', 'channel')`) for hierarchical organization.

### Key Modules

1. **timeseries**: Signal processing, smoothing, derivative estimation, baseline removal
2. **time_slice**: Time-based slicing, state transitions, trial alignment
3. **munge**: Hierarchical DataFrame operations, multi-level indexing utilities
4. **state_space**: Scikit-learn compatible transformers (SoftnormScaler, DataFrameTransformer)
5. **smoothing**: Low-level convolution and Gaussian smoothing
6. **dpca**: Tensor conversion and dPCA analysis for neural data

## Common Patterns

### Working with Time

- Time values are typically `pd.Timedelta` objects
- Use `slice(pd.to_timedelta('0ms'), pd.to_timedelta('500ms'))` for time windows
- Functions accept `timecol` parameter to specify the time index level name

### Hierarchical DataFrames

- Many functions work with multi-level column indices
- Use `hierarchical_assign()` for adding computed columns
- Use `multivalue_xs()` for cross-sections with multiple keys

### Signal Processing

- Smoothing functions accept time-based parameters (e.g., `std=pd.to_timedelta('50ms')`)
- Derivative estimation uses Butterworth or Savitzky-Golay filters
- Functions preserve DataFrame structure and multi-level indices

## Making Changes

### When adding new features:

1. Consider if it fits into an existing module or needs a new one
2. Maintain consistency with existing function signatures and patterns
3. Preserve DataFrame structure and multi-level indices in transformations
4. Add corresponding tests in the appropriate `test_*.py` file
5. Update docstrings with clear parameter descriptions and examples

### When fixing bugs:

1. Add a test case that reproduces the bug
2. Fix the issue with minimal changes
3. Ensure the test passes and no existing tests are broken
4. Consider edge cases related to the fix

### Before committing:

1. Run `black trialframe/` to format code
2. Run `isort trialframe/` to sort imports
3. Run `pytest tests/` to ensure all tests pass
4. Verify changes don't break existing functionality

## Domain-Specific Context

### Neuroscience Experiments

This package is designed for neuroscience and behavioral experiments where:
- Data is organized into discrete trials
- Each trial has time-varying signals (neural activity, kinematics, etc.)
- Trials can be aligned to specific events (stimulus onset, movement start, etc.)
- Different experimental states or epochs need to be tracked within trials

### Common Use Cases

1. **Trial alignment**: Align trials to behavioral events for averaging or comparison
2. **Kinematic analysis**: Compute velocities and accelerations from position data
3. **Neural analysis**: Apply dPCA to separate task-related and condition-independent components
4. **Preprocessing**: Smooth data, remove baselines, normalize signals
5. **Epoch extraction**: Get data from specific time windows or experimental phases

## Important Considerations

- **Performance**: Functions should handle large datasets efficiently (hundreds of trials, high sampling rates)
- **Flexibility**: Support both simple (single-level) and complex (multi-level) DataFrame structures
- **Compatibility**: Maintain scikit-learn transformer compatibility for pipeline integration
- **Data Integrity**: Preserve DataFrame metadata and indices through transformations
