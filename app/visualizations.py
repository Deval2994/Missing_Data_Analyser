"""
Description :   Generates visual comparisons like before/after histograms, boxplots, QQ plots, etc., to help users
                visually validate cleaning methods.

Methods:    visualize_numeric_distribution(df: pd.DataFrame, column_name: str, method: str)
            visualize_categorical_distribution(df: pd.DataFrame, column_name: str, method: str):
"""

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import squarify

def visualize_numeric_distribution(df: pd.DataFrame, column_name: str, method: str):
    """
    Visualizes the distribution of a numeric column from the global DataFrame.

    Parameters:
    - column_name (str): Name of the column in the global df.
    - method (str): One of 'box', 'hist', or 'qq'.
    """
    column = df[column_name]
    fig, ax = plt.subplots(figsize=(6, 4))  # create figure and axes

    if method == "box":
        sns.boxplot(x=column, color="skyblue", ax=ax)
        ax.set_title(f"Box Plot - {column_name}")
        ax.set_xlabel(column_name)

    elif method == "hist":
        sns.histplot(column, kde=True, bins=20, color="orange", edgecolor="black", ax=ax)
        ax.set_title(f"Histogram with PDF - {column_name}")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Frequency")

    elif method == "qq":
        stats.probplot(column, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot - {column_name}")

    else:
        raise ValueError("Invalid method. Use 'box', 'hist', or 'qq'.")

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def visualize_categorical_distribution(df: pd.DataFrame, column_name: str, method: str):
    """
    Visualizes distribution of a categorical or mixed column from the global DataFrame.

    Parameters:
    - column_name (str): Name of the column in the global df.
    - method (str): One of 'pie', 'bar', or 'heatmap'.
    """
    column = df[column_name]
    fig, ax = plt.subplots(figsize=(6, 4))

    if method == 'pie':
        counts = column.value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Pie Chart - {column_name}")
        ax.axis('equal')

    elif method == 'bar':
        counts = column.value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette="pastel", ax=ax)
        ax.set_title(f"Bar Plot - {column_name}")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.5)

    elif method == 'heatmap':
        cross_tab = pd.crosstab(index=df[column_name], columns="count")
        fig, ax = plt.subplots(figsize=(4, len(cross_tab) * 0.4 + 1))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap="YlGnBu", cbar=False, ax=ax)
        ax.set_title(f"Heatmap - Frequency of {column_name}")
        ax.set_ylabel(column_name)

    elif method == 'treemap':
        counts = column.value_counts()
        sizes = counts.values
        labels = [f"{label}\n{count}" for label, count in zip(counts.index, counts.values)]
        fig, ax = plt.subplots(figsize=(6, 4))
        squarify.plot(sizes=sizes, label=labels, alpha=0.8, pad=True, ax=ax, color=sns.color_palette("pastel"))
        ax.set_title(f"Treemap - Frequency of {column_name}")
        ax.axis('off')

    else:
        raise ValueError("Invalid method. Use 'pie', 'bar', or 'heatmap'.")

    fig.tight_layout()
    return fig