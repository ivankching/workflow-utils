import numpy as np
import pandas as pd
import seaborn as sns

def histograms(data: pd.DataFrame):
    data = data.select_dtypes(include=[np.int64, np.float64])
    return data.hist(figsize=(16,20), xlabelsize=7, ylabelsize=7, bins=50)