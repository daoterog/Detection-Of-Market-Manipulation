"""
Benchmarking pipeline module for neural networks.
"""

import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from mynn.layers import Layer
from mynn.neural_network import NeuralNetwork
from mynn.loss import cuadratic_loss
from mynn.utils import train_model, plot_training_history


def get_hidden_layer_units(num_hidden_layers: int, max_num_units: int) -> List[int]:
    """Gets the number of units per hidden layer.
    Args:
        n_hideen_layers (int): Number of hidden layers.
        max_num_units (int): Maximum number of units.
    Returns:
        t.List[int]: List of units per hidden layer.
    """

    def get_combination_list(
        n_layer: int, combinations: List[List[int]]
    ) -> List[List[int]]:
        """Auxiliary function to get the combinations of units per hidden layer.
        Args:
            n_layer (int): Number of hidden layers.
            combinations (t.List[t.List[int]]): List of combinations.
        Returns:
            t.List[t.List[int]]: List of combinations."""
        if n_layer == num_hidden_layers + 1:
            return combinations
        units_per_hidden_layer = list(range(1, max_num_units + 1))
        combinations = [
            combination + [units]
            for combination in combinations
            for units in units_per_hidden_layer
        ]
        return get_combination_list(n_layer + 1, combinations)

    return get_combination_list(1, [[]])


def get_layer_list(
    input_shape: int, output_shape: int, units_per_hidden_layer: List[int]
) -> List[Layer]:
    """Creates a list of layers for a neural network.
    Args:
        input_shape (int): Input shape.
        output_shape (int): Output shape.
        units_per_hidden_layer (t.List[int]): List of units per hidden layer.
    Returns:
        t.List[Layer]: List of layers.
    """
    layers = [Layer(input_shape, units_per_hidden_layer[0], "sigmoid")]
    for idx, units in enumerate(units_per_hidden_layer[:-1]):
        layers.append(Layer(units, units_per_hidden_layer[idx + 1], "sigmoid"))
    layers.append(Layer(units_per_hidden_layer[-1], output_shape, "tanh"))
    return layers


def evaluate_model(
    model: NeuralNetwork,
    train_tuple: Tuple[np.ndarray, np.ndarray],
    val_tuple: Tuple[np.ndarray, np.ndarray],
    test_tuple: Tuple[np.ndarray, np.ndarray],
    verbose: bool = False,
) -> None:
    """Evaluates the model.
    Args:
        model (NeuralNetwork): Neural network.
        train_tuple (t.Tuple[np.ndarray, np.ndarray]): Training data.
        val_tuple (t.Tuple[np.ndarray, np.ndarray]): Validation data.
        test_tuple (t.Tuple[np.ndarray, np.ndarray]): Test data.
        verbose (bool, optional): Verbose. Defaults to False.
        save_results (bool, optional): Save results. Defaults to False.
        save_path (str, optional): Save path. Defaults to "results.json".
    """
    train_x, train_y = train_tuple
    val_x, val_y = val_tuple
    test_x, test_y = test_tuple

    results = {}
    for x_array, y_array, dataset in zip(
        [train_x, val_x, test_x], [train_y, val_y, test_y], ["Train", "Val", "Test"]
    ):
        y_pred = model.predict(x_array)
        loss = cuadratic_loss(y_array.T, y_pred)
        results[dataset] = np.mean(loss)
        if verbose:
            print(f"{dataset} loss: {loss}")
    return results

def make_dir(path: str) -> None:
    """Makes a directory.
    Args:
        path (str): Path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_results(results: dict, path: str) -> None:
    """Saves results to a json file.
    Args:
        results (dict): Results.
        path (str): Path.
    """
    with open(os.path.join(path, 'results.json'), "w", encoding='utf-8') as f:
        json.dump(results, f)

def pipeline(
    lr: List[float],
    num_hidden_layers: List[int],
    max_units: int,
    n_iter: int,
    batch_size: int,
    train_tuple: Tuple[np.ndarray, np.ndarray],
    val_tuple: Tuple[np.ndarray, np.ndarray],
    test_tuple: Tuple[np.ndarray, np.ndarray],
    savepath: str = None,
) -> None:

    """Pipeline for benchmarking.
    Args:
    """

    if savepath is not None:
        make_dir(savepath)

    train_x, train_y = train_tuple

    overall_results_df = pd.DataFrame()

    for learning_rate in lr:
        for n_hidden_layers in num_hidden_layers:
            for combination in get_hidden_layer_units(n_hidden_layers, max_units):
                # Define model name
                model_name = f"lr_{learning_rate}_n_hidden_layers_{n_hidden_layers}_units_{combination}"
                # Create model experiment directory
                model_savepath = os.path.join(savepath, model_name)
                make_dir(model_savepath)
                # Define layers with unit combinations
                layers = get_layer_list(train_x.shape[1], 1, combination)
                # Instance Model
                model = NeuralNetwork(layers,learning_rate, 'cuadratic')
                # Train model
                trained_model, state_history, max_epoch, stop_reason = train_model(model,
                                                                                train_x,
                                                                                train_y,
                                                                                n_iter,
                                                                                batch_size,)
                # Plot training history
                plot_training_history(state_history,
                                    plot_by_epoch=True,
                                    batch_size=batch_size,
                                    train_len=train_x.shape[0],
                                    savefig=True,
                                    savepath=model_savepath,
                                    verbose=False,)
                # Evaluate model
                results = evaluate_model(trained_model,
                                        train_tuple,
                                        val_tuple,
                                        test_tuple)
                # Save results
                results['max_epoch'] = max_epoch
                results['stop_reason'] = stop_reason
                results['num_hidden_layers'] = len(combination)
                results['total_hidden_units'] = sum(combination)
                results['unit_combination'] = combination
                results['learning_rate'] = learning_rate
                save_results(results, model_savepath)
                trained_model.store_parameters(model_savepath)
                # Append results to pandas dataframe
                overall_results_df = overall_results_df.append(results, ignore_index=True)
    # Save overall results
    overall_results_df.to_json(os.path.join(savepath, 'overall_results.json'))
