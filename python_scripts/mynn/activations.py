"""
Activations for neural networks.
"""

import numpy as np


def sigmoid(local_field: np.ndarray) -> np.ndarray:
    """Applies sigmoid function to x."""
    return 1 / (1 + np.exp(-local_field))


def sigmoid_derivative(local_field: np.ndarray) -> np.ndarray:
    """Returns the derivative of the sigmoid function."""
    sigmoid_output = sigmoid(local_field)
    return np.multiply(sigmoid_output, (1 - sigmoid_output))


def linear(local_field: np.ndarray) -> np.ndarray:
    """Applies linear function to x."""
    return local_field


def linear_derivative(local_field: np.ndarray) -> np.ndarray:
    """Returns the derivative of the linear function."""
    return np.ones(local_field.shape)


def hyperbolic_tangent(local_field: np.ndarray) -> np.ndarray:
    """Applies hyperbolic tangent function to x."""
    return np.tanh(local_field)


def hyperbolic_tangent_derivative(local_field: np.ndarray) -> np.ndarray:
    """Returns the derivative of the hyperbolic tangent function."""
    return 1 - np.power(np.tanh(local_field), 2)


def relu(local_field: np.ndarray) -> np.ndarray:
    """Applies relu function to x."""
    return np.maximum(0, local_field)


def relu_derivative(local_field: np.ndarray) -> np.ndarray:
    """Returns the derivative of the relu function."""
    return np.where(local_field > 0, 1, 0)
