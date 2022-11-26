import os
import numpy as np
from mynn.neural_network import NeuralNetwork
from mynn.layers import Layer
from unsupervised_learning.algorithms import MountainClustering, SubstractiveClustering

from mynn.utils import train_model
from helper_functions.data_management import data_loading, joint_random_sampling
from helper_functions.plotting import plot_training_history


def create_autoencoder(num_features: int, num_hidden_units: int, learning_rate: float):

    autoenconder = NeuralNetwork(
        layers=[
            Layer(num_features, num_hidden_units, activation="sigmoid", include_bias=True),
            Layer(num_hidden_units, num_features, activation="sigmoid", include_bias=True),
        ],
        learning_rate=learning_rate,
        loss="cuadratic",
    )

    return autoenconder


def get_autoencoder_embedding(trained_autoencoder: NeuralNetwork, inputs: np.ndarray):

    autoencoder_layers = trained_autoencoder.layers
    encoding_layer = autoencoder_layers[0]

    return encoding_layer.forward(inputs)


def train_autoencoder(manip_features):

    feature_matrices = [
        stock_dict['features'][:, :-2]
        for _, stock_dict in manip_features.items()
    ]

    stacked_feature_matrix = np.vstack(feature_matrices)

    autoencoder = create_autoencoder(num_features=stacked_feature_matrix.shape[1], num_hidden_units=3, learning_rate=1)

    trained_autoencoder, train_history, _, _ = train_model(
        autoencoder, stacked_feature_matrix, stacked_feature_matrix, num_epochs=3, batch_size=1, verbose=True
    )

    plot_training_history(train_history, True, stacked_feature_matrix.shape[0])

    return trained_autoencoder, stacked_feature_matrix.shape[0]


def main(root_folder_path, manip_category, energy_threshold, use_cone):

    # Data Loading
    manip_features = data_loading(root_folder_path, manip_category, energy_threshold, use_cone)

    # Train autoencoder
    trained_autoencoder, num_samples = train_autoencoder(manip_features)

    # Sampling
    percentage_to_sample = 10000 / num_samples
    train_x_all, train_y, val_x_all, val_y, test_x_all, test_y = joint_random_sampling(manip_features, percentage_to_sample, 0.5, 'uniform', False)

    # Eliminating energy feature
    train_x = train_x_all[:, :-1]
    val_x = val_x_all[:, :-1]
    test_x = test_x_all[:, :-1]

    # Get autoencoder embeddings
    train_embedding = get_autoencoder_embedding(trained_autoencoder, train_x.T).T
    val_embedding = get_autoencoder_embedding(trained_autoencoder, val_x.T).T
    test_embedding = get_autoencoder_embedding(trained_autoencoder, test_x.T).T

    # Unsupervised Clustering
    r_a = 1.5
    substractive_clustering = SubstractiveClustering(r_a=r_a,
                                                r_b=1.5*r_a,
                                                distance_metric='euclidean',)
    substractive_clustering.fit(train_x)


if __name__ == "__main__":

    root_folder_path = os.getcwd()
    manip_category = 'poop_and_scoop'
    energy_threshold = 0.5
    use_cone = False

    main(root_folder_path, manip_category, energy_threshold, use_cone)