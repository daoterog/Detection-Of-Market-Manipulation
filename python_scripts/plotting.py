"""
Plotting module.
"""

import os

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def color_plot(
    feature_matrix: np.ndarray,
    freq_index: int,
    modulus_index: int,
    use_cone: bool,
    title: str,
    mask_index: int = None,
    savefig: bool = False,
    figname: str = None,
) -> None:

    """Makes color plot of the specified variable of the feature matrix."""

    if use_cone and mask_index is None:
        raise ValueError("Mask index not specified")

    # Pivot the feature matrix in a way that the complex modulus are mapped to the frequencies in a
    # time coherent manner and sort according to the frequencies in descending fashion.

    # Create dataframe
    if use_cone:
        columns_index = [freq_index, modulus_index, mask_index]
        columns_name = ["frequency", "complex_modulus", "mask"]
    else:
        columns_index = [freq_index, modulus_index]
        columns_name = ["frequency", "complex_modulus"]

    abs_frequency_df = pd.DataFrame(
        feature_matrix[:, columns_index],
        columns=columns_name,
    )

    # Create column with time window
    n_freq = abs_frequency_df.frequency.unique().shape[0]
    n_cols = int(abs_frequency_df.shape[0] / n_freq)
    abs_frequency_df["time_window"] = list(range(n_cols)) * n_freq

    # Create pivot tables with frequency as index and time window as columns
    complex_modulus_pivot = abs_frequency_df.pivot_table(
        index="frequency", columns="time_window", values="complex_modulus"
    )
    complex_modulus_pivot.sort_values(by="frequency", inplace=True, ascending=False)

    if use_cone:
        mask_pivot = abs_frequency_df.pivot_table(
            index="frequency", columns="time_window", values="mask"
        )
        mask_pivot = ~mask_pivot.astype(bool)
        mask_pivot.sort_values(by="frequency", inplace=True, ascending=False)
        mask_pivot.index = np.round(complex_modulus_pivot.index, 2)

    # Round frequencies
    complex_modulus_pivot.index = np.round(complex_modulus_pivot.index, 2)

    # Define xticks
    min_window = complex_modulus_pivot.columns.min()
    max_window = complex_modulus_pivot.columns.max()
    xticks = np.arange(min_window, max_window + 1, 100)

    # Plot
    if use_cone:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), dpi=100)
    else:
        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=100)

    sns.heatmap(complex_modulus_pivot, cmap="viridis", vmin=-1, vmax=1, ax=ax0)
    ax0.set_title("Complete Plot")
    ax0.set_xticks(xticks, labels=xticks)

    if use_cone:
        sns.heatmap(
            complex_modulus_pivot, cmap="viridis", vmin=-1, vmax=1, ax=ax1, mask=mask_pivot
        )
        ax1.set_title("Masked Plot")
        ax1.set_xticks(xticks, labels=xticks)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if savefig:
        if figname is None:
            raise ValueError("Figure name not specified")
        plot_path = os.path.join(ROOT_PATH, "plots", figname)
        fig.savefig(plot_path)


def plot_target_classes(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, stock: str
) -> None:

    """Plot target classes distribution"""

    target_classes, train_occurrences = np.unique(y_train, return_counts=True)
    _, val_occurrences = np.unique(y_val, return_counts=True)
    _, test_occurrences = np.unique(y_test, return_counts=True)
    total_occurrences = train_occurrences + test_occurrences + val_occurrences

    assert (
        train_occurrences.shape[0] >= 2 and test_occurrences.shape[0] >= 2
    ), "Not enough target classes to plot"

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(16, 4))

    width = 1
    color = ["cornflowerblue", "forestgreen"]

    ax0.bar(target_classes, train_occurrences, color=color, width=width, align="center")
    ax0.set_title("Train Absolute\nOccurrences")
    ax0.text(
        1, train_occurrences[1], train_occurrences[1], ha="center", va="bottom", size=15
    )

    ax1.bar(target_classes, val_occurrences, color=color, width=width)
    ax1.set_title("Validation Absolute\nOccurrences")
    ax1.text(
        1, test_occurrences[1], test_occurrences[1], ha="center", va="bottom", size=15
    )

    ax2.bar(target_classes, test_occurrences, color=color, width=width)
    ax2.set_title("Test Absolute\nOccurrences")
    ax2.text(
        1, test_occurrences[1], test_occurrences[1], ha="center", va="bottom", size=15
    )

    ax3.bar(
        target_classes, train_occurrences / total_occurrences, color=color, width=width
    )
    ax3.set_title("Train Relative\nOccurrences")

    ax4.bar(
        target_classes, test_occurrences / total_occurrences, color=color, width=width
    )
    ax4.set_title("Test Relative\nOccurrences")

    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.set_xticks(target_classes)

    fig.suptitle(stock)
    plt.tight_layout()
    plt.show()


def plot3d_dataset(
    feature_matrix: np.ndarray,
    title: str,
    plot_against: str,
    hue_var: str,
    target_var: np.ndarray = None,
) -> None:

    """Generates seaborn pairplot from dataset."""

    if hue_var == "modulus":
        var_index = 5
        var_name = "std_complex_modulus"
    elif hue_var == "target":
        var_index = -1
        var_name = "target"
        if target_var is None:
            raise ValueError("Target variable not specified")
        feature_matrix = np.concatenate(
            (feature_matrix, target_var.reshape(-1, 1)), axis=1
        )
    else:
        raise ValueError(
            "Invalid plot_against argument. Allowed values are 'modulus' or 'target'"
        )

    if plot_against == "frequency":
        index_list = [1, 2, 3, var_index]
    elif plot_against == "time":
        index_list = [0, 2, 3, var_index]
    else:
        raise ValueError(
            "Invalid plot_against argument. Allowed values are 'frequency' or 'time'"
        )

    feature_df = pd.DataFrame(
        feature_matrix[:, index_list],
        columns=[plot_against, "real_part", "imag_part", var_name],
    )

    fig = px.scatter_3d(
        feature_df,
        x=plot_against,
        y="real_part",
        z="imag_part",
        color=var_name,
        size_max=18,
        opacity=0.7,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    fig.update_traces(showlegend=False)
    fig.show()
