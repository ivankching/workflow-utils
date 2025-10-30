import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest, chi2_contingency


def vif(data, exclude=None):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the
    dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to calculate the VIF for.
    exclude : list of str, optional
        A list of feature names to exclude from the calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the VIF for each feature in the dataset.
    """
    if exclude is not None:
        data = data.drop(columns=exclude)
    data = data.select_dtypes(include='number')
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    return vif_data


def kstests(data, label, selected_columns='numeric'):
    """
    Calculate the Kolmogorov-Smirnov test statistic and p-value for each
    feature in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to calculate the test statistic and p-value for.
    label : str
        The label to split the data into two groups [0, 1].
    selected_columns : list or str, optional
        Either 'numeric' to select all numeric features, or a list of
        feature names to select.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the test statistic and p-value for each
        feature in the dataset.
    """
    if label not in data.columns:
        raise ValueError("Label '{}' not found in the dataset".format(label))
    if selected_columns == 'numeric':
        selected_columns = data.select_dtypes(include='number').columns.drop(label)
    zeros = data[data[label] == 0][selected_columns]
    ones = data[data[label] == 1][selected_columns]

    statistics = []
    pvalues = []

    for col in selected_columns:
        if zeros[col].isnull().any() or ones[col].isnull().any():
            raise ValueError("Null values found in the dataset, cannot calculate the test statistic and p-value")
        res = kstest(zeros[col], ones[col])
        statistics.append(res.statistic)
        pvalues.append(res.pvalue)

    return pd.DataFrame({'feature': selected_columns, 'statistic': statistics, 'pvalue': pvalues})

# TODO generate docstring, add unit tests
def pearson_chi2(data, label, selected_columns='category'):
    if label not in data.columns:
        raise ValueError("Label '{}' not found in the dataset".format(label))
    if selected_columns == 'category':
        selected_columns = data.select_dtypes(include=['category', 'object']).columns
        if label in selected_columns:
            selected_columns = selected_columns.drop(label)


    statistics = []
    pvalues = []

    for col in selected_columns:
        contingency_tab = pd.crosstab(data[col], data[label])
        chi2_res = chi2_contingency(contingency_tab)

        # Check if all expected frequencies > 5
        if np.any(chi2_res.expected_freq <= 5):
            statistics.append(None)
            pvalues.append(None)
        else:
            statistics.append(chi2_res.statistic)
            pvalues.append(chi2_res.pvalue)

    return pd.DataFrame({'feature': selected_columns, 'statistic': statistics, 'pvalue': pvalues})

    