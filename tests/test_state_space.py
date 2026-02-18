"""
Tests for the state_space module
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from trialframe.state_space import SoftnormScaler, DataFrameTransformer


def test_softnorm_scaler_fit():
    """Test that SoftnormScaler can be fitted."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    scaler = SoftnormScaler(norm_const=5)
    scaler.fit(X)
    
    # Should have activity_range_ attribute after fitting
    assert hasattr(scaler, 'activity_range_')
    # Range should be max - min for each column
    expected_range = np.array([6, 6, 6])  # 7-1, 8-2, 9-3
    np.testing.assert_array_equal(scaler.activity_range_, expected_range)


def test_softnorm_scaler_transform():
    """Test that SoftnormScaler transforms data correctly."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    scaler = SoftnormScaler(norm_const=5)
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    
    # Shape should be preserved
    assert X_transformed.shape == X.shape
    # Values should be scaled by (range + norm_const)
    expected = X / (scaler.activity_range_ + 5)
    np.testing.assert_array_almost_equal(X_transformed, expected)


def test_softnorm_scaler_fit_transform():
    """Test that SoftnormScaler can fit and transform in one step."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    scaler = SoftnormScaler(norm_const=5)
    X_transformed = scaler.fit_transform(X)
    
    assert X_transformed.shape == X.shape
    # Should have same result as separate fit and transform
    scaler2 = SoftnormScaler(norm_const=5)
    scaler2.fit(X)
    X_transformed2 = scaler2.transform(X)
    np.testing.assert_array_almost_equal(X_transformed, X_transformed2)


def test_softnorm_scaler_with_nan():
    """Test SoftnormScaler with NaN values."""
    X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    scaler = SoftnormScaler(norm_const=5)
    scaler.fit(X)
    
    # nanmax and nanmin should ignore NaN
    assert not np.isnan(scaler.activity_range_[1])
    
    X_transformed = scaler.transform(X)
    # NaN should be preserved
    assert np.isnan(X_transformed[1, 1])


def test_softnorm_scaler_different_norm_const():
    """Test SoftnormScaler with different normalization constants."""
    X = np.array([[0, 0], [10, 10]])
    
    scaler1 = SoftnormScaler(norm_const=0)
    scaler1.fit(X)
    X1 = scaler1.transform(X)
    
    scaler2 = SoftnormScaler(norm_const=10)
    scaler2.fit(X)
    X2 = scaler2.transform(X)
    
    # With larger norm_const, scaled values should be smaller (more regularization)
    # For the non-zero row, X2 should be less than X1
    assert np.all(X2[1] < X1[1])


def test_softnorm_scaler_zero_range():
    """Test SoftnormScaler when a feature has zero range."""
    X = np.array([[1, 5], [1, 10], [1, 15]])
    scaler = SoftnormScaler(norm_const=5)
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    
    # First column has zero range, should be divided by norm_const only
    np.testing.assert_array_almost_equal(X_transformed[:, 0], X[:, 0] / 5)


def test_dataframe_transformer_fit():
    """Test that DataFrameTransformer can be fitted."""
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
    })
    
    pca = PCA(n_components=2)
    transformer = DataFrameTransformer(transformer=pca)
    transformer.fit(X)
    
    # Should have is_fitted_ attribute
    assert hasattr(transformer, 'is_fitted_')
    assert transformer.is_fitted_ is True
    # Underlying transformer should also be fitted
    assert hasattr(transformer.transformer, 'components_')


def test_dataframe_transformer_transform():
    """Test that DataFrameTransformer preserves DataFrame structure."""
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
    }, index=pd.Index(['r1', 'r2', 'r3', 'r4', 'r5']))
    
    pca = PCA(n_components=2)
    transformer = DataFrameTransformer(transformer=pca)
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    
    # Should return a DataFrame
    assert isinstance(X_transformed, pd.DataFrame)
    # Index should be preserved
    assert X_transformed.index.equals(X.index)
    # Should have correct number of components
    assert X_transformed.shape[1] == 2


def test_dataframe_transformer_fit_transform():
    """Test DataFrameTransformer fit_transform."""
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
        'c': [3, 6, 9, 12, 15],
    })
    
    pca = PCA(n_components=2)
    transformer = DataFrameTransformer(transformer=pca)
    X_transformed = transformer.fit_transform(X)
    
    # Should return a DataFrame
    assert isinstance(X_transformed, pd.DataFrame)
    # Index should be preserved
    assert X_transformed.index.equals(X.index)
    # Should have correct number of components
    assert X_transformed.shape[1] == 2


def test_dataframe_transformer_with_scaler():
    """Test DataFrameTransformer with SoftnormScaler."""
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50],
    })
    
    scaler = SoftnormScaler(norm_const=5)
    transformer = DataFrameTransformer(transformer=scaler)
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    
    # Should return a DataFrame
    assert isinstance(X_transformed, pd.DataFrame)
    # Index should be preserved
    assert X_transformed.index.equals(X.index)
    # Columns should be numbered (0, 1)
    assert X_transformed.shape[1] == 2


def test_dataframe_transformer_preserves_multiindex():
    """Test that DataFrameTransformer preserves MultiIndex."""
    idx = pd.MultiIndex.from_product(
        [[1, 2], ['a', 'b']],
        names=['trial_id', 'state']
    )
    X = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [2, 4, 6, 8],
    }, index=idx)
    
    pca = PCA(n_components=2)
    transformer = DataFrameTransformer(transformer=pca)
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    
    # Index should be preserved including MultiIndex
    assert isinstance(X_transformed.index, pd.MultiIndex)
    assert X_transformed.index.equals(X.index)


def test_dataframe_transformer_different_output_shape():
    """Test DataFrameTransformer when output has different number of columns."""
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
        'c': [3, 6, 9, 12, 15],
        'd': [4, 8, 12, 16, 20],
    })
    
    # Reduce to 2 components
    pca = PCA(n_components=2)
    transformer = DataFrameTransformer(transformer=pca)
    X_transformed = transformer.fit_transform(X)
    
    # Should have 2 columns now instead of 4
    assert X_transformed.shape == (5, 2)
    # Columns should be numbered 0, 1
    assert list(X_transformed.columns) == [0, 1]
