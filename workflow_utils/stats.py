import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest


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

def kstest(data, label, columns='numeric'):
    """
    Calculate the Kolmogorov-Smirnov test statistic and p-value for each
    feature in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to calculate the test statistic and p-value for.
    label : str
        The label to split the data into two groups [0, 1].
    columns : list or str, optional
        Either 'numeric' to select all numeric features, or a list of
        feature names to select.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the test statistic and p-value for each
        feature in the dataset.
    """
    if columns is 'numeric':
        columns = data.select_dtypes(include='number').columns
    zeros = data[data[label] == 0][columns]
    ones = data[data[label] == 1][columns]

    statistics = []
    pvalues = []

    for col in columns:
        res = kstest(zeros[col], ones[col])
        statistics.append(res.statistic)
        pvalues.append(res.pvalue)

    return pd.DataFrame({'feature': columns, 'statistic': statistics, 'pvalue': pvalues})