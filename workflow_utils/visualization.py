import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil

def histograms(data: pd.DataFrame, size: tuple = (16,20), bins: int = 50):
    """Return histograms of all numeric fields in a grid"""
    data = data.select_dtypes(include=[np.int64, np.float64])
    return data.hist(figsize=size, xlabelsize=7, ylabelsize=7, bins=bins)

def kdes(data: pd.DataFrame, size: tuple = (16,20)):
    """Return KDE plots of all numeric fields stacked vertically"""
    data = data.select_dtypes(include=[np.int64, np.float64])
    return data.plot.kde(figsize=size, subplots=True)

def countplots(data: pd.DataFrame, cols: int = 6, size: tuple = (16,20)):
    """Return countplots figure of all categorical fields"""
    data = data.select_dtypes(exclude=[np.int64, np.float64])
    cols = min(cols, data.shape[1])
    rows = ceil(data.shape[1] / cols)
    num_features = len(data.columns)
    fig = plt.figure()
    ax = fig.subplots(rows, cols)

    if cols == 1 and rows == 1:
        sns.countplot(x=data.iloc[:,0])
    elif rows == 1:
        for i in range(num_features):
            sns.countplot(x=data[data.columns[i]], ax=ax[int(i%cols)])
    elif cols == 1:
        for i in range(num_features):
            sns.countplot(x=data[data.columns[i]], ax=ax[int(i/cols)])
    else:
        for i in range(num_features):
            sns.countplot(x=data[data.columns[i]], ax=ax[int(i/cols)][int(i%cols)])
            
    if num_features < rows*cols:
        # remove empty subplots
        for i in range(num_features, rows*cols):
            ax[int(i/cols)][int(i%cols)].remove()
    fig.set_size_inches(*size)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def correlation_heatmap(data):
    """Return heatmap of pairwise correlations for all numeric features"""
    data = data.select_dtypes(include=[np.int64, np.float64])
    corr = data.corr()
    heatmap = sns.heatmap(corr, cmap="YlGnBu", annot=True)
    return heatmap