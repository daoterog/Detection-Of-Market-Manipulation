from mynn.neural_network import NeuralNetwork
from mynn.layers import Layer
from mynn.utils import train_model
from helper_functions.data_management import modulus_loading
from helper_functions.plotting import plot_training_history


def create_autoencoder(num_features: int, num_hidden_units: int):

    autoenconder = NeuralNetwork(
        layers=[
            Layer(num_features, num_hidden_units, activation="sigmoid"),
            Layer(num_hidden_units, num_features, activation="sigmoid"),
        ],
        learning_rate=0.1,
        loss="cuadratic",
    )

    return autoenconder


def main():

    manip_category = "pump_and_dump"
    stock_name = "srpt"

    manip_features = modulus_loading(manip_category)

    modulus_matrix = manip_features[stock_name]['modulus']

    autoencoder = create_autoencoder(modulus_matrix.shape[0], 30)

    trained_model, train_history, _, _ = train_model(
        autoencoder, modulus_matrix.T, modulus_matrix.T, num_epochs=20, batch_size=1, verbose=True
    )

    plot_training_history(train_history, True, modulus_matrix.shape[1])

if __name__ == "__main__":
    main()
