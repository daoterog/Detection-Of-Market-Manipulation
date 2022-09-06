"""
Contains first functions used in the project.
"""

import os

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import norm, nbinom
from sklearn.utils import shuffle


def load_excel_data(folder_path: str) -> Dict[str, np.ndarray]:
    """Loads data from the data folder and store them in a dictionary.
    Args:
        folder_path (str): Path to the folder containing the data.
    Returns:
        Dict[str, np.ndarray]: Dictionary with dataframes of stock belonging to the category.
    """
    data_filenames = os.listdir(folder_path)
    stocks_dict = {}
    for filename in data_filenames:
        coef_and_freq = pd.read_excel(
            os.path.join(folder_path, filename), sheet_name="cwt", header=0, dtype=str
        ).to_numpy()
        stocks_dict[filename.split("_")[0]] = coef_and_freq
    return stocks_dict


def standardize_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Standardize the matrix.
    Args:
        feature_matrix (np.ndarray): Matrix to standardize.
    Returns:
        np.ndarray: Standardized matrix.
    """
    feat_min = np.min(feature_matrix, axis=0)
    feat_max = np.max(feature_matrix, axis=0)
    return (feature_matrix - feat_min) / (feat_max - feat_min)


def build_feature_matrix(stock_matrix: np.ndarray) -> dict:
    """Builds feature matrix with time, frequencies, real and imaginary parts of the coefficients,
        and the complex modulus of the coefficients.
    Args:
        stock_matrix (np.ndarray): matrix with data.
    Returns:
    """

    stock_copy = stock_matrix.copy()

    # Get cone of influence and delete it from matrix
    cone_of_influence = stock_copy[0, :-1].astype(float)
    stock_copy = np.delete(stock_copy, 0, axis=0)

    # Get frequencies and delete them from matrix
    frequencies = stock_copy[:, -1].astype(float)
    stock_copy = np.delete(stock_copy, -1, axis=1)

    # Convert coefficients into complex dtype
    stock_array = stock_copy
    stock_array = np.char.replace(stock_array, "i", "j")
    stock_array = np.char.replace(stock_array, " ", "")
    stock_array = stock_array.astype(np.complex128)

    # Get real and imaginary coefficients
    real_coefficients = np.real(stock_array)
    imag_coefficients = np.imag(stock_array)

    # Map frequencies to coefficients
    sample_list = []

    for i_freq in range(frequencies.shape[0]):
        for j_time in range(real_coefficients.shape[1]):
            sample = np.array(
                [
                    j_time,
                    frequencies[i_freq],
                    real_coefficients[i_freq, j_time],
                    imag_coefficients[i_freq, j_time],
                ]
            )
            sample_list.append(sample)

    # Create feature matrix
    feature_matrix = np.hstack(sample_list)

    # Get complex modulus and append it to the feature matrix
    real_power = np.power(feature_matrix[:, 2], 2)
    imag_power = np.power(feature_matrix[:, 3], 2)
    complex_modulus = np.sqrt(real_power + imag_power).reshape(-1, 1)
    feature_matrix = np.concatenate((feature_matrix, complex_modulus), axis=1)

    return feature_matrix, cone_of_influence


def assign_labels(
    feature_matrix: np.ndarray,
    energy_threshold: float,
    cone_of_influence: np.ndarray = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Assigns labels to the feature matrix.
    Args:
        feature_matrix (np.ndarray): Feature matrix.
        energy_threshold (float): Threshold for the energy of the coefficients.
        cone_of_influence (np.ndarray): Cone of influence.
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Feature matrix with labels and mask for plotting
            valid energy points.
    """

    # Assign labels to samples according to a threshold
    theshold_labels = feature_matrix[:, 5] > 2 * energy_threshold - 1

    if cone_of_influence is not None:

        # Standardize cone of influence for it to be suitable to compare with the frequencies
        std_cone_of_influence = standardize_matrix(cone_of_influence)

        # Append repeated cone of influence to the feature matrix
        n_freq = len(np.unique(feature_matrix[:, 1]))
        repeated_cone_of_influence = np.array(std_cone_of_influence.tolist() * n_freq)

        # Compare frequencies with stndardize cone to determine which samples are suitable for
        # prediction
        mask = feature_matrix[:, 1] > repeated_cone_of_influence
        real_labels = theshold_labels & mask
        mask = mask.reshape(-1, 1)
        real_labels = real_labels.astype(int).reshape(-1, 1)
        feature_matrix = np.concatenate((feature_matrix, real_labels), axis=1)

        return feature_matrix, mask

    theshold_labels = theshold_labels.reshape(-1, 1)
    feature_matrix = np.concatenate((feature_matrix, theshold_labels), axis=1)

    return feature_matrix, None


def data_loading(
    manip_category: str,
    energy_threshold: float,
    use_cone: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Loads the data.
    Args:
        manip_category (str): Name of the manipulation category.
        energy_threshold (float): Threshold for the energy of the coefficients.
        use_cone (bool): Whether to use the cone of influence or not.
    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary with numpy of stock belonging to the
            category.
    """

    # Get data folder path
    root_folder_path = os.path.dirname(os.getcwd())
    manip_folder_path = os.path.join(root_folder_path, "data", manip_category)

    # Load raw data
    stocks_dict = load_excel_data(manip_folder_path)

    # Preprocess data
    stocks_features = {}
    for stock_name, stock_matrix in stocks_dict.items():
        # Get features
        feature_matrix, cone_of_influence = build_feature_matrix(stock_matrix)
        # Standardize features
        feature_matrix = standardize_matrix(feature_matrix)
        # Change modulus range
        feature_matrix[:, -1] = feature_matrix[:, -1] * 2 - 1
        # Assign labels
        if use_cone:
            feature_matrix, mask = assign_labels(
                feature_matrix, energy_threshold, cone_of_influence
            )
        else:
            feature_matrix, mask = assign_labels(feature_matrix, energy_threshold)
        # Append to dictionary
        stocks_features[stock_name] = {"features": feature_matrix, "mask": mask}

    return stocks_features


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
) -> pd.DataFrame:

    """Samples the data according to certain probability distribution.

    Args:
        feature_matrix (np.ndarray): dataset to sample from.
        distribution (str): distribution to use to generate the random numbers.
        train_size (float): size of the training set.

    Returns:
        (pd.DataFrame, pd.DataFrame): train and test datasets."""

    shuffled_array = shuffle(feature_matrix)

    n_iter = int(np.ceil(shuffled_array.shape[0] * train_size))

    if distribution == "normal":
        random_number_array = np.random.normal(0, 1, size=(n_iter,))
    elif distribution == "negative_binomial":
        random_number_array = np.random.negative_binomial(1, 0.1, size=(n_iter,))
    elif distribution == "uniform":
        percentile_list = np.random.uniform(0, 1, size=(n_iter,))
    else:
        raise ValueError(
            "Distribution not recognized. Available distributions are: normal, negative_binomial, uniform"
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


def random_sampling(
    feature_matrix: np.ndarray, train_size: float, val_size: float, distribution: str
) -> None:

    """
    Randomly samples data from the feature matrix.
    """

    # Get number of unique frequencies
    num_freqs = np.unique(feature_matrix[:, 1]).shape[0]
    stop_cut = feature_matrix.shape[0]
    n_samples = int(stop_cut / num_freqs)

    # Define list to store the samples
    train_samples = []
    val_samples = []
    test_samples = []

    # Loop through all frequencies
    for start_cut in range(0, stop_cut, n_samples):

        # Cut and shuffle array
        freq_array = feature_matrix[start_cut : start_cut + n_samples, :]
        shuffled_array = shuffle(freq_array)

        # Sanity check for correct sampling
        assert np.unique(freq_array[:, 1]).shape[0] == 1, "Frequencies are not unique"

        # Sample train data
        train_array, test_array = sample_data(shuffled_array, distribution, train_size)

        # Sample test data
        val_array, test_array = sample_data(test_array, distribution, val_size)

        train_samples.append(train_array)
        val_samples.append(val_array)
        test_samples.append(test_array)

    train_array = np.vstack(train_samples)
    val_array = np.vstack(val_samples)
    test_array = np.vstack(test_samples)

    X_train = train_array[:, :-1]
    y_train = train_array[:, -1]

    X_val = val_array[:, :-1]
    y_val = val_array[:, -1]

    X_test = test_array[:, :-1]
    y_test = test_array[:, -1]

    return X_train, y_train, X_val, y_val, X_test, y_test


def joint_random_sampling(
    manip_stock_dict: dict, train_size: float, val_size: float, distribution: str
) -> tuple:

    """Performs random sampling on the feature matrix of the manipulated stocks and then stacks them
    together."""

    # List to store sampled feature matrices
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    X_test_list = []
    y_test_list = []

    # Loop over stock dictionary
    for _, stock_features in manip_stock_dict.items():

        # Perform random sampling over specific stock and append the results to the list
        (
            stock_X_train,
            stock_y_train,
            stock_X_val,
            stock_y_val,
            stock_X_test_all,
            stock_y_test,
        ) = random_sampling(
            stock_features["feature_matrix"], train_size, val_size, distribution
        )
        X_train_list.append(stock_X_train)
        y_train_list.append(stock_y_train)
        X_val_list.append(stock_X_val)
        y_val_list.append(stock_y_val)
        X_test_list.append(stock_X_test_all)
        y_test_list.append(stock_y_test)

    # Stack the feature matrices
    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test_all = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return X_train_all, y_train, X_val_all, y_val, X_test_all, y_test
