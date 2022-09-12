"""
Experimentation module
"""

import os
import random
import numpy as np

from helper_functions.data_management import data_loading, joint_random_sampling
from mynn.benchmarking import pipeline

np.random.seed(777)
random.seed(777)

def get_experimentation_results(
    manip_category: str, energy_threshold: float, use_cone: bool
) -> None:
    """Get experimentation results.
    Args:
        manip_category (str): Manipulation category.
        energy_threshold (float): Energy threshold.
        use_cone (bool): Use cone.
    """
    # Load data
    manip_features = data_loading(manip_category, energy_threshold, use_cone)
    # Split data
    (
        train_x_all,
        _,
        val_x_all,
        _,
        test_x_all,
        _,
    ) = joint_random_sampling(manip_features, 0.6, 0.5, "uniform", undersample=True)
    # Characteristics filtering
    train_tuple = (train_x_all[:, :4], train_x_all[:, 4].reshape(-1, 1))
    val_tuple = (val_x_all[:, :4], val_x_all[:, 4].reshape(-1, 1))
    test_tuple = (test_x_all[:, :4], test_x_all[:, 4].reshape(-1, 1))
    # Define number of iterations
    n_iter = int(np.ceil(train_x_all.shape[0] * 50 / BATCH_SIZE))
    # Define savepath
    root_path = os.path.join(os.getcwd(), "./../nn_results", manip_category)
    # Run pipeline
    pipeline(
        LEARNING_RATES,
        NUM_HIDDEN_LAYERS,
        MAX_UNITS,
        n_iter,
        BATCH_SIZE,
        train_tuple,
        val_tuple,
        test_tuple,
        savepath=root_path,
    )

LEARNING_RATES = [0.2, 0.5, 0.9]
NUM_HIDDEN_LAYERS = [1, 2, 3]
MAX_UNITS = 5
BATCH_SIZE = 32

if __name__ == "__main__":

    categories = ["pump_and_dump", "poop_and_scoop", "wash_trade"]
    thresholds = [0.75] * 3
    cone_flags = [True, False, False]

    for category, threshold, cone_flag in zip(categories, thresholds, cone_flags):
        get_experimentation_results(category, threshold, cone_flag)
