"""
Metrics module.
"""

import numpy as np


def cosine_distance(feature_matrix: np.ndarray, datapoint: np.ndarray) -> np.ndarray:
    """Calculates the cosine distance between two data points.
    Args:
        feature_matrix (np.ndarray): feature matrix.
        datapoint (np.ndarray): data point.
    Returns:
        np.ndarray: Cosine distance between the feature_matrix and datapoint.
    """
    return 1 - np.dot(feature_matrix, datapoint.T) / (
        np.linalg.norm(feature_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(datapoint)
    )


def euclidean_distance(feature_matrix: np.ndarray, datapoint: np.ndarray) -> np.ndarray:
    """Calculates the euclidean distance between two data points.
    Args:
        feature_matrix (np.ndarray): feature matrix.
        datapoint (np.ndarray): data point.
    Returns:
        np.ndarray: Euclidean distance between the feature_matrix and datapoint.
    """
    datapoint = datapoint.reshape(-1, 1).T
    return np.linalg.norm(feature_matrix - datapoint, axis=1).reshape(-1, 1)


def manhattan_distance(feature_matrix: np.ndarray, datapoint: np.ndarray) -> np.ndarray:
    """Calculates the manhattan distance between two data points.
    Args:
        feature_matrix (np.ndarray): feature matrix.
        datapoint (np.ndarray): data point.
    Returns:
        np.ndarray: Manhattan distance between the feature_matrix and datapoint.
    """
    datapoint = datapoint.reshape(-1, 1).T
    return np.sum(np.abs(feature_matrix - datapoint), axis=1).reshape(-1, 1)


def mahalanobis_distance(
    first_datapoint: np.ndarray,
    second_datapoint: np.ndarray,
    inverse_covariance_matrix: np.ndarray,
) -> np.ndarray:
    """Calculates the mahalanobis distance between two data points.
    Args:
        feature_matrix (np.ndarray): feature matrix.
        datapoint (np.ndarray): data point.
        inverse_covariance_matrix (np.ndarray): Inverse covariance matrix.
    Returns:
        np.ndarray: Mahalanobis distance between the two data points.
    """
    first_datapoint = first_datapoint.reshape(-1, 1)
    second_datapoint = second_datapoint.reshape(-1, 1)
    return np.sqrt(
        np.dot(
            np.dot((first_datapoint - second_datapoint).T, inverse_covariance_matrix),
            (first_datapoint - second_datapoint),
        )
    )


def build_mahalanobis_distance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Builds the mahalanobis distance matrix.
    Args:
        feature_matrix (np.ndarray): Feature matrix.
    Returns:
        np.ndarray: Mahalanobis distance matrix.
    """
    covariance_matrix = np.cov(feature_matrix.T, ddof=1)
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    distance_matrix = [
        [
            mahalanobis_distance(
                first_datapoint, second_datapoint, inverse_covariance_matrix
            )
            for second_datapoint in feature_matrix
        ]
        for first_datapoint in feature_matrix
    ]

    return np.array(distance_matrix).squeeze().T


def get_distance_matrix(feature_matrix: np.ndarray, distance_metric: str) -> np.ndarray:
    """Calculates the distance matrix between all data points.
    Args:
        feature_matrix (np.ndarray): Feature matrix.
        distance_metric (str): Distance metric.
    Returns:
        np.ndarray: Distance matrix.
    """

    if distance_metric == "cosine":
        distance_criterion = cosine_distance
    elif distance_metric == "euclidean":
        distance_criterion = euclidean_distance
    elif distance_metric == "manhattan":
        distance_criterion = manhattan_distance
    elif distance_metric == "mahalanobis":
        return build_mahalanobis_distance_matrix(feature_matrix)
    else:
        raise ValueError(
            "Invalid distance metric. Please choose from: cosine, euclidean,"
            + " manhattan, mahalanobis."
        )

    distance_list = [
        distance_criterion(feature_matrix, datapoint) for datapoint in feature_matrix
    ]

    return np.concatenate(distance_list, axis=1)
