"""
My own neural network.
"""

import typing as t

import numpy as np

from activations import (
    sigmoid,
    sigmoid_derivative,
    linear,
    linear_derivative,
    hyperbolic_tangent,
    hyperbolic_tangent_derivative,
    relu,
    relu_derivative,
)

np.random.seed(777)


class Layer:

    """Layers object of neural network."""

    def __init__(
        self, input_size: np.ndarray, output_size: np.ndarray, activation: str, include_bias: bool = False
    ):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.include_bias = include_bias
        if self.include_bias:
            self.bias = np.zeros((output_size, 1))

        self.inputs = None
        self.local_field = None
        self.outputs = None

        self.local_gradient = None
        self.weights_grad = None
        self.bias_grad = None

        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "tanh":
            self.activation = hyperbolic_tangent
            self.activation_derivative = hyperbolic_tangent_derivative
        elif activation == "linear":
            self.activation = linear
            self.activation_derivative = linear_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError(
                "Invalid activation function. Please choose between 'sigmoid', "
                + "'hyperbolic_tangent' and 'linear'."
            )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Makes the forward pass of the layer.
        Args:
            inputs (np.ndarray): Inputs of the layer.
        Returns:
            np.ndarray: Outputs of the layer.
        """
        self.inputs = inputs
        self.local_field = np.matmul(self.weights, inputs)
        if self.include_bias:
            self.local_field = self.local_field + self.bias
        self.outputs = self.activation(self.local_field)
        return self.outputs

    def backward(self, previous_local_gradient: np.ndarray) -> np.ndarray:
        """Makes the backward pass of the layer.
        Args:
            error (np.ndarray): Error of the layer.
            learning_rate (float): Learning rate of the neural network.
        Returns:
            np.ndarray: Local gradient of the layer.
        """
        self.local_gradient = np.multiply(
            previous_local_gradient, self.activation_derivative(self.local_field)
        )
        self.weights_grad = np.matmul(self.local_gradient, self.inputs.T)
        if self.include_bias:
            self.bias_grad = np.mean(self.local_gradient, axis=1, keepdims=True)
        return np.matmul(self.weights.T, self.local_gradient)

    def step(self, learning_rate: float) -> None:
        """Updates the weights and bias of the layer.
        Args:
            learning_rate (float): Learning rate of the neural network.
        """
        self.weights = self.weights - learning_rate * self.weights_grad
        if self.include_bias:
            self.bias = self.bias - learning_rate * self.bias_grad

    def get_grads(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Returns the gradients of the layer.
        Returns:
            t.Tuple[np.ndarray, np.ndarray]: Gradients of the layer.
        """
        return self.weights_grad, self.bias_grad
