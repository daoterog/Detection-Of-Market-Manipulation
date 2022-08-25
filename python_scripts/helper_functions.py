"""
Contains first functions used in the project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    labels = feature_matrix[:, 4] > energy_threshold
    labels = labels.astype(int)
    feature_matrix = np.concatenate((feature_matrix, labels.reshape(-1, 1)), axis=1)

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
        stocks_features[stock] = build_feature_matrix(stocks_dict[stock], energy_threshold)

    return stocks_features


def color_plot(
    feature_matrix: np.ndarray,
    var_index: int,
    title: str,
    ax: plt.Axes,
) -> None:

    """Makes color plot of the specified variable of the feature matrix."""

    # Pivot the feature matrix in a way that the complex modulus are mapped to the frequencies in a
    # time coherent manner and sort according to the frequencies in descending fashion.
    abs_frequency_df = pd.DataFrame(
        feature_matrix[:, [0, var_index]], columns=["frequency", 'complex_modulus']
    )
    n_freq = abs_frequency_df.frequency.unique().shape[0]
    n_cols = int(abs_frequency_df.shape[0] / n_freq)
    abs_frequency_df["time_window"] = list(range(n_cols)) * n_freq
    abs_frequency_df = abs_frequency_df.pivot_table(
        index="frequency", columns="time_window", values='complex_modulus'
    )
    abs_frequency_df.sort_values(by="frequency", inplace=True, ascending=False)

    # Define xticks
    min_window = abs_frequency_df.columns.min()
    max_window = abs_frequency_df.columns.max()
    xticks = np.arange(min_window, max_window + 1, 100)

    # Plot
    sns.heatmap(abs_frequency_df, cmap="viridis", vmin=-1, vmax=1, ax=ax)
    ax.set_xticks(xticks, labels=xticks)
    ax.set_title(title)
    plt.show()
