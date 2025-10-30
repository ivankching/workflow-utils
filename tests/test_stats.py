import seaborn as sns
import pandas as pd
import numpy as np
from workflow_utils import stats
from scipy.stats import kstest, chi2_contingency
import pytest


def test_vif():
    data = stats.vif(sns.load_dataset('iris'), exclude=['species'])
    assert data.shape == (4,2)

    import pandas as pd


def test_kstests_with_numeric_columns():
    data = pd.DataFrame({
        'label': [0, 1, 0, 1, 0, 1],
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [7, 8, 9, 10, 11, 12],
        'feature3': [13, 14, 15, 16, 17, 18]
    })
    result = stats.kstests(data, 'label')
    assert result.shape == (3, 3)

def test_kstests_with_selected_columns():
    data = pd.DataFrame({
        'label': [0, 1, 0, 1, 0, 1],
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [7, 8, 9, 10, 11, 12],
        'feature3': [13, 14, 15, 16, 17, 18]
    })
    result = stats.kstests(data, 'label', selected_columns=['feature1', 'feature3'])
    assert result.shape == (2, 3)

def test_kstests_with_missing_label():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [7, 8, 9, 10, 11, 12],
        'feature3': [13, 14, 15, 16, 17, 18]
    })

    with pytest.raises(ValueError):
        stats.kstests(data, 'nonexistent_label')

def test_kstests_with_null_values():
    data = pd.DataFrame({
        'label': [0, 1, 0, 1, 0, 1],
        'feature1': [1, 2, 3, np.nan, 5, 6],
        'feature2': [7, 8, 9, 10, 11, 12],
        'feature3': [13, 14, 15, 16, 17, 18]
    })
    with pytest.raises(ValueError):
        stats.kstests(data, 'label')

from workflow_utils.stats import pearson_chi2

def test_pearson_chi2():
    # Test case 1: Check if the function raises a ValueError when the label is not found in the dataset
    data = pd.DataFrame({'label': [0, 1], 'feature': [1, 2]})
    with pytest.raises(ValueError):
        stats.pearson_chi2(data, 'nonexistent_label')

    # Test case 2: Check if the function returns the correct result for titanic dataset
    data = sns.load_dataset('titanic')
    result = stats.pearson_chi2(data, 'alive')
    assert result.shape == (6, 3)