"""
Contains first functions used in the project.
"""

import os
import pandas as pd
import numpy as np

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


def build_feature_matrix(stock_df: pd.DataFrame) -> dict:

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

    stock_features = {
        "feature_matrix": np.stack(sample_list),
        "cone_of_influence": cone_of_influence,
    }

    return stock_features


def get_feature_matrices(stocks_dict: dict) -> dict:

    """Builds feature matrices of all the stocks."""

    stocks_feature_matrices = {}

    for stock in stocks_dict.keys():
        stocks_feature_matrices[stock] = build_feature_matrix(stocks_dict[stock])

    return stocks_feature_matrices

def data_loading(sheet_name: str) -> dict:

    """Loads the data."""

    stocks_dict = load_excel_data(sheet_name)
    stocks_features = get_feature_matrices(stocks_dict)

    return stocks_features
