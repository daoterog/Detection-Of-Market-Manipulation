"""
Utils module.
"""

import os
import typing as t
from collections import OrderedDict
from csv import reader

import pandas as pd
import numpy as np
import scipy.io as sio
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from scipy.stats import norm, nbinom

from mynn.neural_network import NeuralNetwork


def get_batch(
    feature_matrix: np.ndarray, targets: np.ndarray, batch_size: int
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Returns a batch of the feature matrix and targets.
    Args:
        feature_matrix (np.ndarray): Transposed feature matrix.
        targets (np.ndarray): Transposed targets.
    Returns:
        t.Tuple[np.ndarray, np.ndarray]: Batch of the feature matrix and targets.
    """
    for i in range(0, feature_matrix.shape[1], batch_size):
        yield feature_matrix[:, i : i + batch_size], targets[:, i : i + batch_size]


def get_mean_epoch_gradient(
    grad_list: t.List[t.Dict[int, np.ndarray]]
) -> t.Dict[int, np.ndarray]:
    """Gets the mean gradient for each epoch.
    Args:
        grad_list (t.List[t.Dict[int, np.ndarray]]): List of gradients.
    Returns:
        t.Dict[int, np.ndarray]: Mean gradient for each epoch.
    """
    mean_grads = OrderedDict()
    for layer in grad_list[0].keys():
        grad_layer = []
        for iter_grads in grad_list:
            grad_layer.append(iter_grads[layer].reshape(-1, 1))
        mean_grads[layer] = np.mean(np.hstack(grad_layer), axis=0)
    return mean_grads


def differentiate_gradients(
    diff_grads: t.List[t.Dict[int, np.ndarray]]
) -> t.List[float]:
    """Differentiates the gradients.
    Args:
        diff_grads (t.List[t.Dict[int, np.ndarray]]): Differentiated gradients.
    Returns:
        t.List[float]: Differentiated gradients.
    """
    if len(diff_grads) > 2:
        diff_grads.pop(0)
    return [
        np.abs(prev_grad - cur_grad)
        for prev_grad, cur_grad in zip(diff_grads[0].values(), diff_grads[1].values())
    ]


def evaluate_early_stopping_criteria(
    loss_list: t.List[int],
    weight_grad_list: t.List[t.Dict[int, np.ndarray]],
    weight_diff_grads: t.List[t.Dict[int, np.ndarray]],
    bias_grad_list: t.List[t.Dict[int, np.ndarray]],
    bias_diff_grads: t.List[t.Dict[int, np.ndarray]],
    previous: bool,
) -> t.Tuple[bool, t.Optional[str], t.Optional[str]]:
    """Evaluates if the early stopping criteria are met.
    Args:
    """

    # Evaluate Loss convergence
    if np.mean(loss_list) <= 1e-6:
        return True, "Loss converged", None
    # Evaluate Gradient convergence
    weight_diff_grads.append(get_mean_epoch_gradient(weight_grad_list))
    if not any(bias_grad_list):
        if len(weight_diff_grads) > 1:
            weight_diff = differentiate_gradients(weight_diff_grads)
            if np.all(np.array(weight_diff) < 1e-5):
                if previous:
                    return True, "Weight gradients converged", None
                else:
                    return False, None, True
    else:
        bias_diff_grads.append(get_mean_epoch_gradient(bias_grad_list))
        if len(weight_diff_grads) > 1 and len(bias_diff_grads) > 1:
            weight_diff = differentiate_gradients(weight_diff_grads)
            bias_diff = differentiate_gradients(bias_diff_grads)
            if np.all(np.array(weight_diff) < 1e-5) and np.all(
                np.array(bias_diff) < 1e-5
            ):
                if previous:
                    return True, "Weight and bias gradients converged", None
                else:
                    return False, None, True

    return False, None, False


def train_model(
    model: NeuralNetwork,
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    num_epochs: int,
    batch_size: int,
    verbose: bool = False,
) -> t.Tuple[NeuralNetwork, OrderedDict, int, str]:
    """Trains the model for a given number of epochs and returns the training history.
    Args:
        model (NeuralNetwork): Neural network model.
        feature_matrix (np.ndarray): Feature matrix.
        targets (np.ndarray): Targets.
        n_iter (int): Number of iterations.
        batch_size (int): Batch size.
        verbose (bool, optional): Whether to print the loss. Defaults to False.
    Returns:
        t.Tuple[NeuralNetwork, OrderedDict, int, str]: Trained model, training history, epoch in
            which the algorithm stopped, and reason for stopping.
    """
    train_history = OrderedDict()
    iter_num = 0
    weight_diff_grads = []
    bias_diff_grads = []
    previous = False
    stop_criterion = None
    for epoch in range(num_epochs):
        loss_list = []
        weight_grad_list = []
        bias_grad_list = []
        for batch_inputs, batch_targets in get_batch(
            feature_matrix.T, targets.T, batch_size
        ):
            # Forward pass
            _ = model.forward(batch_inputs)
            # Compute loss
            loss = np.mean(model.get_loss(batch_targets))
            loss_list.append(loss)
            # Backward pass
            model.backward(batch_targets)
            # Update parameters
            model.step()
            # Get grads
            weight_grads, bias_grads = model.get_grads()
            # Store state
            train_history[iter_num] = {
                "loss": loss,
                "weight_grads": weight_grads,
                "bias_grads": bias_grads,
            }
            weight_grad_list.append(weight_grads)
            bias_grad_list.append(bias_grads)
            iter_num += 1
        # Evaluate verbosity
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {np.mean(loss_list)}")

        if len(weight_diff_grads) > 2:
            weight_diff_grads.pop(0)

        if len(bias_diff_grads) > 2:
            bias_diff_grads.pop(0)

        stop, stop_criterion, previous = evaluate_early_stopping_criteria(loss_list,
                                                                        weight_grad_list,
                                                                        weight_diff_grads,
                                                                        bias_grad_list,
                                                                        bias_diff_grads,
                                                                        previous)
        if stop:
            print("Stopping...")
            print("Stopping Criterion: ", stop_criterion)
            break

    if stop_criterion is None:
        stop_criterion = "Max epochs reached"

    return model, train_history, epoch, stop_criterion


def unstack_grads(
    grads: t.Tuple[t.Dict[int, np.ndarray]]
) -> t.Tuple[t.Dict[int, t.List[np.ndarray]], t.Dict[int, t.List[float]]]:
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
    mean_grads: t.Dict[int, t.List[float]]
) -> t.Dict[int, np.ndarray]:
    """Stacks the mean gradients.
    Args:
        mean_grads (t.Dict[int, t.List[float]]): Mean gradients.
    Returns:
        t.Dict[int, np.ndarray]: Stacked mean gradients.
    """
    return np.stack(list(mean_grads.values())).T


def plot_training_history(
    train_history: OrderedDict,
    plot_by_epoch: bool = False,
    batch_size: int = None,
    train_len: int = None,
    verbose: bool = True,
    savefig: bool = False,
    savepath: str = ".",
) -> None:
    """Plots the training history.
    Args:
        train_history (OrderedDict): Training history.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        savepath (str, optional): Path to save the figure. Defaults to ".".
    """

    # Unzip items
    iters, loss, weight_grads, bias_grads = zip(
        *[
            (k, v["loss"], v["weight_grads"], v["bias_grads"])
            for k, v in train_history.items()
        ]
    )

    # Unstack gradients
    _, weight_layer_mean_grads = unstack_grads(weight_grads)
    _, bias_layer_mean_grads = unstack_grads(bias_grads)

    include_bias = any(bias_grads)

    # Stack mean gradients
    weight_stacked_mean_grads = stack_mean_gradients(weight_layer_mean_grads)
    if include_bias:
        bias_stacked_mean_grads = stack_mean_gradients(bias_layer_mean_grads)

    xlabel = "Iteration"
    if plot_by_epoch:
        mean_loss = np.array(loss).reshape(-1, 1)
        weight_stacked_mean_grads, mean_loss = zip(
            *[
                (np.mean(grad_batch, axis=1), np.mean(loss_batch))
                for grad_batch, loss_batch in get_batch(
                    weight_stacked_mean_grads.T, mean_loss.T, train_len//batch_size
                )
            ]
        )
        iters = list(range(len(mean_loss[:-1])))
        xlabel = "Epoch"
        if include_bias:
            bias_stacked_mean_grads = [
                np.mean(grad_batch)
                for grad_batch, _ in get_batch(
                    bias_stacked_mean_grads.T, bias_stacked_mean_grads.T, train_len
                )
            ]

    if include_bias:
        _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 3))
    else:
        _, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3))

    ax0.plot(iters, mean_loss[:-1])
    ax0.set_title("Loss")
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel("Loss")

    ax1.plot(iters, weight_stacked_mean_grads[:-1])
    ax1.set_title("Mean weight gradients")
    ax1.legend([f"Layer {i}" for i in range(len(weight_stacked_mean_grads[0]))])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Mean gradient")

    if include_bias:
        ax2.plot(iters, bias_stacked_mean_grads[:-1])
        ax2.set_title("Mean bias gradients")
        ax2.legend([f"Layer {i}" for i in range(len(bias_stacked_mean_grads[0]))])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Mean gradient")

    plt.tight_layout()
    if savefig:
        path = os.path.join(savepath, "training_history.png")
        plt.savefig(path)
    if verbose:
        plt.show()


def readmatfile(filepath: str) -> np.ndarray:
    """Reads a .mat file.
    Args:
        filepath (str): filepath.
    Returns:
        np.ndarray: Data.
    """
    # Read matlab file
    mat = sio.loadmat(filepath)
    # Define column names
    columns = list(mat.keys() - ["__header__", "__version__", "__globals__"])
    # Iterate over columns length and extract the most repeated length
    all_n_rows = []
    for column in columns:
        n_rows = mat[column].shape[0]
        if n_rows != 1:
            all_n_rows.append(n_rows)
    row_len = np.unique(all_n_rows, return_counts=True)[0][0]
    # Store columns that have the most repeated length in a list and stack them into a ndarray
    col_list = []
    for column in columns:
        if mat[column].shape[0] == row_len:
            col_list.append(mat[column])
    return np.hstack(col_list)


def readfile(filename: str) -> np.ndarray:
    """Reads a csv file and returns a numpy array.
    Args:
        filename (str): Name of the file.
    Returns:
        np.ndarray: data.
    """
    filepath = os.path.join(os.getcwd(), "data", filename)
    if filename[-3:] == "mat":
        return readmatfile(filepath)

    with open(filepath, "r", encoding="UTF-8") as file:
        csv_reader = reader(file)
        data = np.vstack(list(csv_reader)).astype(np.float)
    return data


def binary_search_percentile(
    random_number: float,
    distribution: str,
    lower_bound: float,
    upper_bound: float,
    tol: float,
) -> float:

    """Recieves a random number generated from a specific distribution and uses binary search to
    find the percentile to which it corresponds.

    Args:
        random_number (float): random number generated from a specific distribution.
        distribution (str): name of the used distribution.
        lower_bound (float): lower bound of the percentile search.
        upper_bound (float): upper bound of the percentile search.
        tol (float): tol of the percentile search.

    Returns:
        float: percentile of the random number."""

    middle_percentile = (lower_bound + upper_bound) / 2

    if distribution == "normal":
        percentile_number = norm.ppf(middle_percentile, loc=0, scale=1)
    else:
        percentile_number = nbinom.ppf(middle_percentile, n=1, p=0.1)

    if abs(random_number - percentile_number) < tol:
        return middle_percentile
    elif random_number > percentile_number:
        return binary_search_percentile(
            random_number, distribution, middle_percentile, upper_bound, tol
        )
    else:
        return binary_search_percentile(
            random_number, distribution, lower_bound, middle_percentile, tol
        )


def sample_data(
    feature_matrix: np.ndarray,
    distribution: str,
    train_size: float,
) -> t.Tuple[np.ndarray, np.ndarray]:

    """Samples the data according to certain probability distribution.

    Args:
        feature_matrix (np.ndarray): dataset to sample from.
        distribution (str): distribution to use to generate the random numbers.
        train_size (float): size of the training set.

    Returns:
        t.Tuple[np.ndarray, np.ndarray]: train and test datasets."""

    shuffled_array = shuffle(feature_matrix)

    n_iter = int(np.ceil(shuffled_array.shape[0] * train_size))

    if distribution == "normal":
        random_number_array = np.random.normal(0, 1, size=(n_iter,))
    elif distribution == "nbinom":
        random_number_array = np.random.negative_binomial(1, 0.1, size=(n_iter,))
    elif distribution == "uniform":
        percentile_list = np.random.uniform(0, 1, size=(n_iter,))
    else:
        raise ValueError(
            "Distribution not recognized. Available distributions are: normal, nbinom,"
            + " uniform"
        )

    # Find percentile for each random number
    if distribution == "normal" or distribution == "negative_binomial":
        percentile_list = [
            binary_search_percentile(random_number, distribution, 0.0, 1.0, 10e-3)
            for _, random_number in enumerate(random_number_array)
        ]

    train_samples = []

    # Draw samples from the shuffled array according to the percentiles
    for _, percentile in enumerate(percentile_list):
        sample_index = int(percentile * shuffled_array.shape[0])
        train_samples.append(shuffled_array[sample_index, :])
        shuffled_array = np.delete(shuffled_array, sample_index, axis=0)
        shuffled_array = shuffle(shuffled_array)

    train_array = np.vstack(train_samples)
    test_array = shuffled_array

    return train_array, test_array


def standardize_data(data: np.ndarray) -> np.ndarray:
    """Standardizes the data.
    Args:
        data (np.ndarray): Data to normalize.
    Returns:
        np.ndarray: Standardized data.
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_std = (data - data_min) / (data_max - data_min)
    return data_std * 2 - 1


def plot3d_data(data: np.ndarray, indices: list) -> None:
    """Plots 3D data by selecting the variables contained in the indices list.
    Args:
        data (np.ndarray): Data to plot.
        indices (list): List of indices of the variables to plot.
    """
    feature_df = pd.DataFrame(data)
    fig = px.scatter_3d(
        feature_df,
        x=indices[0],
        y=indices[1],
        z=indices[2],
        size_max=18,
        opacity=0.7,
    )
    fig.update_traces(showlegend=False)
    fig.show()
