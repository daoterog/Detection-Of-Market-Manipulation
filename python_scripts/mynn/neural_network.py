"""
Neural Network module.
"""

import os
import typing as t
import json

import numpy as np

from mynn.layers import Layer
from mynn.loss import (
    cuadratic_loss,
    cuadratic_loss_derivative,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
)


class NeuralNetwork:

    """Neural Network object."""

    def __init__(self, layers: t.List[Layer], learning_rate: float, loss: str):
        self.layers = layers
        self.learning_rate = learning_rate

        if loss == "cuadratic":
            self.loss = cuadratic_loss
            self.loss_derivative = cuadratic_loss_derivative
        elif loss == "binary_cross_entropy":
            self.loss = binary_cross_entropy
            self.loss_derivative = binary_cross_entropy_derivative
        else:
            raise ValueError("Loss function not found.")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation.
        Args:
            inputs (np.ndarray): Input data.
        Returns:
            np.ndarray: Output data.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, y_true: np.ndarray) -> None:
        """Backward propagation.
        Args:
            y_true (np.ndarray): True labels.
        """
        previous_local_gradient = self.loss_derivative(y_true, self.layers[-1].outputs)
        for layer in reversed(self.layers):
            previous_local_gradient = layer.backward(previous_local_gradient)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predicts the output of the neural network.
        Args:
            inputs (np.ndarray): Input data.
        Returns:
            np.ndarray: Output data.
        """
        return self.forward(inputs.T)

    def step(self) -> None:
        """Makes a step of the neural network."""
        for layer in self.layers:
            layer.step(self.learning_rate)

    def get_loss(self, y_true: np.ndarray) -> float:
        """Gets the loss of the neural network.
        Args:
            y_true (np.ndarray): True labels.
        Returns:
            float: Loss of the neural network.
        """
        return self.loss(y_true, self.layers[-1].outputs)

    def get_grads(self) -> t.Tuple[t.Dict[int,np.ndarray], t.Dict[int,np.ndarray]]:
        """Gets the gradients of the weights and bias of the neural network.
        Returns:
            t.Tuple[t.Dict[np.ndarray], t.Dict[np.ndarray]]: Gradients of the weights and bias.
        """
        weight_grads = {}
        bias_grads = {}
        for lay_num, layer in enumerate(self.layers):
            weight_grad, bias_grad = layer.get_grads()
            weight_grads[lay_num] =  weight_grad
            if layer.include_bias:
                bias_grads[lay_num] = bias_grad
        return weight_grads, bias_grads

    def store_parameters(self, path: str) -> None:
        """Stores the parameters of the neural network.
        Args:
            path (str): Path where the parameters will be stored.
        """
        weight_dict = {}
        bias_dict = {}
        for lay_num, layer in enumerate(self.layers):
            layer_weights, layer_bias = layer.get_parameters()
            weight_dict[lay_num] = layer_weights
            if layer.include_bias:
                bias_dict[lay_num] = layer_bias

        weights_path = os.path.join(path, "weights.json")
        with open(weights_path, "w", encoding='utf-8') as file:
            json.dump(weight_dict, file)

        biases_path = os.path.join(path, "biases.json")
        with open(biases_path, "w", encoding='utf-8') as file:
            json.dump(bias_dict, file)

    def load_parameters(self, path: str) -> None:
        """Loads the parameters of the neural network.
        Args:
            path (str): Path where the parameters are stored.
        """
        weights_path = os.path.join(path, "weights.json")
        with open(weights_path, "r", encoding='utf-8') as file:
            weight_dict = json.load(file)

        biases_path = os.path.join(path, "biases.json")
        with open(biases_path, "r", encoding='utf-8') as file:
            bias_dict = json.load(file)

        for lay_num, layer in enumerate(self.layers):
            layer_weights = weight_dict.get(str(lay_num))
            layer_bias = bias_dict.get(str(lay_num))
            layer.set_parameters(layer_weights, layer_bias)
