import numpy as np
import pandas as pd
import seaborn as sns

def histograms(data: pd.DataFrame, size: tuple = (16,20), bins: int = 50):
    data = data.select_dtypes(include=[np.int64, np.float64])
    return data.hist(figsize=size, xlabelsize=7, ylabelsize=7, bins=bins)