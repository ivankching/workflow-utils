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

def plot_proportions(data):
    """
    Plot stacked horizontal bars of categories over years, with
    each bar colored according to its category.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with years as index and categories as columns
        whose values are the proportions of each category in each year

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and Axes objects of the plot
    """
    data = data.round(1)
    years = data.index
    categories = data.columns
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps["RdYlGn"](np.linspace(0.15, 0.85, data.shape[1]))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for colname, color in zip(categories, category_colors):
        widths = data[colname]
        starts = data_cum[colname] - widths
        rects = ax.barh(years, widths, left=starts, height=0.5, label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(categories), bbox_to_anchor=(0, -.1), loc='lower left', fontsize='small')
    return fig, ax

def labeled_lineplot(series, offset=0.3):
    """
    Plot a lineplot of a series with its values labeled on the
    line. The labels are offset from the line by a specified
    amount.

    Parameters
    ----------
    series : pd.Series
        Series to plot
    offset : float, optional
        Offset of the labels from the line. Default is 0.3.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object of the plot
    """
    ax = sns.lineplot(series)
    for x, y in zip(series.index, series):
        plt.text(x=x,
                y=y-offset,
                s='{:.1f}'.format(y),
                color='purple')
    return ax