"""
Plotting module.
"""

import os

from typing import Dict, Tuple, List
from collections import OrderedDict

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def color_plot(
    stock_dict: Dict[str, np.ndarray],
    title: str,
    savefig: bool = False,
    figname: str = None,
) -> None:
    """Makes color plot.
    Args:
        stock_dict (Dict[str,np.ndarray]): Dictionary with stock names as keys and
            feature matrices as values.
        title (str): Title of the plot.
        savefig (bool): Whether to save the figure or not.
        figname (str): Name of the figure to be saved.
    """

    # Instance parameters
    feature_matrix = stock_dict["features"]
    mask = stock_dict["mask"]

    # Create dataframe
    abs_frequency_df = pd.DataFrame(
        feature_matrix[:, [0, 1, 4]],
        columns=["time_window", "frequency", "complex_modulus"],
    )

    # Pivot the feature matrix in a way that the complex modulus are mapped to the frequencies in a
    # time coherent manner and sort according to the frequencies in descending fashion.

    # Create pivot tables with frequency as index and time window as columns
    complex_modulus_pivot = abs_frequency_df.pivot_table(
        index="frequency", columns="time_window", values="complex_modulus"
    )
    complex_modulus_pivot.sort_values(by="frequency", inplace=True, ascending=False)

    if mask is not None:
        abs_frequency_df["mask"] = mask
        mask_pivot = abs_frequency_df.pivot_table(
            index="frequency", columns="time_window", values="mask"
        )
        mask_pivot = ~mask_pivot.astype(bool)
        mask_pivot.sort_values(by="frequency", inplace=True, ascending=False)
        mask_pivot.index = np.round(complex_modulus_pivot.index, 2)
        mask_pivot.columns = np.round(complex_modulus_pivot.columns, 2)

    # Round frequencies
    complex_modulus_pivot.index = np.round(complex_modulus_pivot.index, 2)
    complex_modulus_pivot.columns = np.round(complex_modulus_pivot.columns, 2)

    # Plot
    if mask is not None:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), dpi=100)
    else:
        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=100)

    sns.heatmap(complex_modulus_pivot, cmap="viridis", vmin=-1, vmax=1, ax=ax0)
    ax0.set_title("Complete Plot")

    if mask is not None:
        sns.heatmap(
            complex_modulus_pivot,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            ax=ax1,
            mask=mask_pivot,
        )
        ax1.set_title("Masked Plot")

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
    """Plot target classes distribution.
    Args:
        y_train (np.ndarray): Training target classes.
        y_val (np.ndarray): Validation target classes.
        y_test (np.ndarray): Test target classes.
        stock (str): Stock name.
    """

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

    ax1.bar(target_classes, val_occurrences, color=color, width=width)
    ax1.set_title("Validation Absolute\nOccurrences")

    ax2.bar(target_classes, test_occurrences, color=color, width=width)
    ax2.set_title("Test Absolute\nOccurrences")

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
) -> None:
    """Generates seaborn pairplot from dataset.
    Args:
        feature_matrix (np.ndarray): Feature matrix.
        title (str): Title of the plot.
        plot_against (str): Variable to plot against.
        hue_var (str): Variable to use for hue."""

    if hue_var == "modulus":
        var_index = 4
        var_name = "std_complex_modulus"
    elif hue_var == "target":
        var_index = -1
        var_name = "target"
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


def plot_umap(
    embedding: np.ndarray, target: np.ndarray, n_components: int = 2, title: str = ""
) -> None:
    """Plots UMAP embedding of feature matrix.
    Args:
        feature_matrix (np.ndarray): Feature matrix.
        target (np.ndarray): Target vector.
        n_neighbors (int, optional): Number of neighbors. Defaults to 15.
        min_dist (float, optional): Minimum distance. Defaults to 0.1.
        n_components (int, optional): Number of components. Defaults to 2.
        metric (str, optional): Distance metric. Defaults to 'euclidean'.
        title (str, optional): Plot title. Defaults to ''.
    """
    if n_components == 2:
        fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], color=target)
    elif n_components == 3:
        fig = px.scatter_3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], color=target
        )
    else:
        raise ValueError("n_components must be 2 or 3")
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
    fig.show()


def unstack_grads(
    grads: Tuple[Dict[int, np.ndarray]]
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[float]]]:
    """Unstacks the gradients.
    Args:
        grads (t.Tuple[t.Dict[int, np.ndarray]]): Gradients.
    Returns:
        t.Tuple[t.Dict[int, t.List[np.ndarray]], t.Dict[int, t.List[float]]]: Tuple of unstacked
            gradients, the first one containing the whole gradient matrices and the second one the
            mean of the gradients.
    """
    layer_grads = OrderedDict()
    layer_mean_grads = OrderedDict()
    for iter_grads in grads:
        for layer, grad in iter_grads.items():
            if layer not in layer_grads:
                layer_grads[layer] = []
                layer_mean_grads[layer] = []
            layer_grads[layer].append(grad)
            layer_mean_grads[layer].append(np.mean(grad))
    return layer_grads, layer_mean_grads


def stack_mean_gradients(
    mean_grads: Dict[int, List[float]]
) -> Dict[int, np.ndarray]:
    """Stacks the mean gradients.
    Args:
        mean_grads (t.Dict[int, t.List[float]]): Mean gradients.
    Returns:
        t.Dict[int, np.ndarray]: Stacked mean gradients.
    """
    return np.stack(list(mean_grads.values())).T


def get_batch(
    feature_matrix: np.ndarray, targets: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a batch of the feature matrix and targets.
    Args:
        feature_matrix (np.ndarray): Transposed feature matrix.
        targets (np.ndarray): Transposed targets.
    Returns:
        t.Tuple[np.ndarray, np.ndarray]: Batch of the feature matrix and targets.
    """
    for i in range(0, feature_matrix.shape[1], batch_size):
        yield feature_matrix[:, i : i + batch_size], targets[:, i : i + batch_size]


def plot_training_history(
    train_history: OrderedDict,
    plot_by_epoch: bool = False,
    train_len: int = None,
    savefig: bool = False,
    figname: str = "model_history",
) -> None:
    """Plots the training history.
    Args:
        train_history (OrderedDict): Training history.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        figname (str, optional): Figure name. Defaults to 'model_history'.
    """

    # Unzip items
    iters, loss, weight_grads, bias_grads = zip(
        *[
            (k, v["loss"], v["weight_grads"], v["bias_grads"])
            for k, v in train_history.items()
        ]
    )

    # Get mean loss
    mean_loss = [np.mean(loss_i) for loss_i in loss]

    # Unstack gradients
    _, weight_layer_mean_grads = unstack_grads(weight_grads)
    _, bias_layer_mean_grads = unstack_grads(bias_grads)

    include_bias = any(bias_grads)

    # Stack mean gradients
    weight_stacked_mean_grads = stack_mean_gradients(weight_layer_mean_grads)
    if include_bias:
        bias_stacked_mean_grads = stack_mean_gradients(bias_layer_mean_grads)

    xlabel = 'Iteration'
    if plot_by_epoch:
        mean_loss = np.array(mean_loss).reshape(-1, 1)
        print(mean_loss.shape)
        weight_stacked_mean_grads, mean_loss = zip(*[
            (np.mean(grad_batch, axis=1), np.mean(loss_batch, axis=1))
            for grad_batch, loss_batch in get_batch(weight_stacked_mean_grads.T, mean_loss.T, train_len)
        ])
        iters = list(range(len(mean_loss)))
        xlabel = 'Epoch'
        if include_bias:
            bias_stacked_mean_grads = [
                np.mean(grad_batch)
                for grad_batch, _ in get_batch(bias_stacked_mean_grads.T, bias_stacked_mean_grads.T, train_len)
            ]

    if include_bias:
        _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 3))
    else:
        _, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3))

    ax0.plot(iters, mean_loss)
    ax0.set_title("Loss")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel("Loss")

    ax1.plot(iters, weight_stacked_mean_grads)
    ax1.set_title("Mean weight gradients")
    ax1.legend([f"Layer {i}" for i in range(len(weight_stacked_mean_grads[0]))])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Mean gradient")

    if include_bias:
        ax2.plot(iters, bias_stacked_mean_grads)
        ax2.set_title("Mean bias gradients")
        ax2.legend([f"Layer {i}" for i in range(len(bias_stacked_mean_grads[0]))])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Mean gradient")

    plt.tight_layout()
    if savefig:
        plt.savefig(f"{figname}.png")
    plt.show()
