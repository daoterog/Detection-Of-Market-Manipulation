"""
Contains first functions used in the project.
"""

import os
import pandas as pd

ROOT_FOLDER_PATH = os.path.dirname(os.getcwd())
DATA_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, 'data')
DATA_FILE_NAMES = os.listdir(DATA_FOLDER_PATH)
TRANS_SHEETS = ['stf_spectogram', 'cwt', 'cwt_gaussian', 'pwr_spectrum', 'spectogram']

def data_loading() -> dict:

    """
    Loads data from the data folder and store them in a dictionary.

    Returns:
        dict: Dictionary with dataframes.
    """

    stocks_dict = {}

    for filename in DATA_FILE_NAMES:
        transforms_dict = {}

        if filename not in set(['gme_coefficients.xlsx', 'nok_coefficients.xlsx']):
            for sheet_name in TRANS_SHEETS:

                coef_and_freq = pd.read_excel(os.path.join(DATA_FOLDER_PATH, filename), header=1)

                if sheet_name != 'spectogram':
                    coefficents = ['coef_' + str(i) for i in range(coef_and_freq.shape[1] - 1)]
                    frequencies = ['freq']

                    if sheet_name == 'cwt_gaussian':
                        coefficents = coefficents[:-1]
                        frequencies.append('scales')

                    coef_and_freq.columns = coefficents + frequencies

                transforms_dict[sheet_name] = coef_and_freq

            stocks_dict[filename.split('_')[0]] = transforms_dict

    return stocks_dict

if __name__ == '__main__':
    data_loading()
