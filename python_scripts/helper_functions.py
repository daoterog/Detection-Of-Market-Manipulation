"""
Contains first functions used in the project.
"""

import os

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from scipy.stats import norm, nbinom
from sklearn.utils import shuffle

ROOT_FOLDER_PATH = os.path.dirname(os.getcwd())
DATA_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "data")
DATA_FILE_NAMES = os.listdir(DATA_FOLDER_PATH)


def load_excel_data(sheet_name: str) -> dict:

    """
    Loads data from the data folder and store them in a dictionary.

    Args:
        sheet_name (str): sheet name to read from excel.

    Returns:
        dict: Dictionary with dataframes.
    """

    stocks_dict = {}

    for filename in DATA_FILE_NAMES:

        coef_and_freq = pd.read_excel(
            os.path.join(DATA_FOLDER_PATH, filename),
            sheet_name=sheet_name,
            header=0,
        )

        if sheet_name != "spectogram":
            coefficents = [
                "coef_" + str(i + 1) for i in range(coef_and_freq.shape[1] - 1)
            ]
            frequencies = ["freq"]

            if sheet_name == "cwt_gaussian":
                coefficents = coefficents[:-1]
                frequencies.append("scales")

            coef_and_freq.columns = coefficents + frequencies

        stocks_dict[filename.split("_")[0]] = coef_and_freq

    return stocks_dict


def build_feature_matrix(stock_df: pd.DataFrame, energy_threshold: float) -> dict:

    """Rearanges coefficient values and maps them to its frequency."""

    stock_copy = stock_df.copy()

    cone_of_influence = stock_copy.iloc[0, :-1].astype(float).to_numpy()
    stock_copy.drop(index=[0], inplace=True)

    frequencies = stock_copy.iloc[:, -1].astype(float)
    stock_copy.drop(columns=["freq"], inplace=True)

    if frequencies.unique().shape[0] != frequencies.shape[0]:
        frequencies = frequencies.to_frame().reset_index()

    frequencies = frequencies.to_numpy()

    stock_array = stock_copy.to_numpy().astype(str)
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
                    frequencies[i_freq],
                    real_coefficients[i_freq, j_time],
                    imag_coefficients[i_freq, j_time],
                ]
            )
            sample_list.append(sample)

    # Create feature matrix
    feature_matrix = np.stack(sample_list)

    # Get complex modulus and append it to the feature matrix
    real_power = np.power(feature_matrix[:, 1], 2)
    imag_power = np.power(feature_matrix[:, 2], 2)
    complex_modulus = np.sqrt(real_power + imag_power).reshape(-1, 1)
    feature_matrix = np.concatenate((feature_matrix, complex_modulus), axis=1)

    # Standardize complex modulus and append it to the feature matrix
    modulus_complex = feature_matrix[:, 3]
    min_modulus_complex = modulus_complex.min()
    max_modulus_complex = modulus_complex.max()
    std_modulus_complex = (modulus_complex - min_modulus_complex) / (
        max_modulus_complex - min_modulus_complex
    )
    std_modulus_complex = std_modulus_complex * 2 - 1
    feature_matrix = np.concatenate(
        (feature_matrix, std_modulus_complex.reshape(-1, 1)), axis=1
    )

    # Assign labels to samples according to a threshold
    theshold_labels = feature_matrix[:, 4] > energy_threshold

    # Filter samples with energy lower than cone of influence

    # Standardize cone of influence for it to be suitable to compare with the frequencies
    min_cone = cone_of_influence.min()
    max_cone = cone_of_influence.max()
    std_cone_of_influence = (cone_of_influence - min_cone) / (max_cone - min_cone)
    min_freq = frequencies.min()
    max_freq = frequencies.max()
    std_cone_of_influence = std_cone_of_influence * (max_freq - min_freq) + min_freq

    # Append repeated cone of influence to the feature matrix
    n_freq = frequencies.shape[0]
    repeated_cone_of_influence = np.array(std_cone_of_influence.tolist() * n_freq)

    # Compare frequencies with stndardize cone to determine which samples are suitable for
    # prediction
    is_valid = feature_matrix[:, 0] > repeated_cone_of_influence
    real_labels = theshold_labels & is_valid
    is_valid = is_valid.reshape(-1, 1)
    real_labels = real_labels.astype(int).reshape(-1, 1)
    feature_matrix = np.concatenate((feature_matrix, is_valid, real_labels), axis=1)

    stock_features = {
        "feature_matrix": feature_matrix,
        "cone_of_influence": cone_of_influence,
    }

    return stock_features


def data_loading(sheet_name: str, energy_threshold: float) -> dict:

    """Loads the data."""

    stocks_dict = load_excel_data(sheet_name)

    stocks_features = {}
    for stock in stocks_dict.keys():
        stocks_features[stock] = build_feature_matrix(
            stocks_dict[stock], energy_threshold
        )

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


def random_sampling(
    feature_matrix: np.ndarray, train_size: float, distribution: str
) -> None:

    """
    Randomly samples data from the feature matrix.
    """

    # Get number of unique frequencies
    num_freqs = np.unique(feature_matrix[:, 0]).shape[0]
    stop_cut = feature_matrix.shape[0]
    n_samples = int(stop_cut / num_freqs)
    n_train = int(n_samples * train_size)

    # Define list to store the samples
    train_samples = []
    test_samples = []

    # Loop through all frequencies
    for start_cut in range(0, stop_cut, n_samples):

        # Cut and shuffle array
        freq_array = feature_matrix[start_cut : start_cut + n_samples, :]
        shuffled_array = shuffle(freq_array)

        # Sanity check for correct sampling
        assert np.unique(freq_array[:, 0]).shape[0] == 1, "Frequencies are not unique"

        # Generate list of random numbers according to distribution
        if distribution == "normal":
            random_number_array = np.random.normal(0, 1, size=(n_train,))
        elif distribution == "negative_binomial":
            random_number_array = np.random.negative_binomial(1, 0.1, size=(n_train,))
        elif distribution == "uniform":
            percentile_list = np.random.uniform(0, 1, size=(n_train,))
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

        # Draw samples from the shuffled array according to the percentiles
        for _, percentile in enumerate(percentile_list):
            sample_index = int(percentile * shuffled_array.shape[0])
            train_samples.append(shuffled_array[sample_index, :])
            shuffled_array = np.delete(shuffled_array, sample_index, axis=0)

        test_samples.append(shuffled_array)

    train_array = np.vstack(train_samples)
    test_array = np.vstack(test_samples)

    X_train = train_array[:, :-1]
    y_train = train_array[:, -1]

    X_test = test_array[:, :-1]
    y_test = test_array[:, -1]

    return X_train, y_train, X_test, y_test
