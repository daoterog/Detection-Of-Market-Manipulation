"""
Losses module.
"""

import numpy as np


def cuadratic_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns the loss function.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        np.ndarray: Loss function.
    """
    return np.power((y_pred - y_true), 2) / 2


def cuadratic_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns the derivative of the loss function.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        np.ndarray: Derivative of the loss function.
    """
    return y_pred - y_true


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns the binary cross entropy loss function.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        np.ndarray: Binary cross entropy loss function.
    """
    y_pred_zero = np.where(np.isclose(y_pred, 0), y_pred, np.log(y_pred))
    y_pred_one = np.where(np.isclose(y_pred, 1), 1 - y_pred, np.log(1 - y_pred))
    return - y_true * y_pred_zero - (1 - y_true) * y_pred_one


def binary_cross_entropy_derivative(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Returns the derivative of the binary cross entropy loss function.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        np.ndarray: Derivative of the binary cross entropy loss function.
    """
    epsilon = 1e-7
    y_pred_zero = np.where(y_pred == 0, epsilon, y_pred)
    y_pred_one = np.where(y_pred == 1, 1 - epsilon, y_pred)
    return (y_pred - y_true) / (y_pred_zero * (1 - y_pred_one))
