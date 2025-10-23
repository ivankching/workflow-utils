import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
