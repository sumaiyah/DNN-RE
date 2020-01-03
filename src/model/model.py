MODEL_ACTIVATIONS_PATH = 'misc_data/'
TRAIN_DATA_PATH = 'misc_data/XOR_train_data.csv'
TEST_DATA_PATH = 'misc_data/XOR_test_data.csv'
MODEL_PATH = 'misc_data/XOR.h5'
CLASS_ENCODINGS = {0: 'Zero', 1: 'One'}

import pandas as pd
from typing import List
import keras.models as keras


def compute_layerwise_activations(model: keras.Model, data_path: str):
    """
    Store sampled activations for each layer in CSV files
    """

    data_x = pd.read_csv(data_path, usecols=(lambda column : column not in ['target']))

    n_layers = len(model.layers)
    for layer_index in range(0, n_layers):
        partial_model = keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)

        # e.g. h_1_0, h_1_1, ...
        neuron_labels = ['h_' + str(layer_index) + '_' + str(i)
                         for i in range(0, model.layers[layer_index].output_shape[1])]

        activation_values = pd.DataFrame(data=partial_model.predict(data_x), columns=neuron_labels)
        activation_values.to_csv(MODEL_ACTIVATIONS_PATH + str(layer_index) + '.csv', index=False)

class Model:
    """
    Represent trained neural network model
    """

    def __init__(self, recompute_layer_activations=False):
        self.model: keras.Model = keras.load_model(MODEL_PATH)
        self.n_layers = len(self.model.layers)

        self.class_encodings = CLASS_ENCODINGS


        if recompute_layer_activations:
            compute_layerwise_activations(self.model, TRAIN_DATA_PATH)

    def get_layer_activations(self, layer_index: int):
        """
        Return activation values given layer index
        """
        filename = MODEL_ACTIVATIONS_PATH + str(layer_index) + '.csv'
        return pd.read_csv(filename)

    def get_layer_activations_of_neuron(self, layer_index: int, neuron_index: int):
        """
        Return activation values given layer index, only return the column for a given neuron index
        """
        filename = MODEL_ACTIVATIONS_PATH + str(layer_index) + '.csv'
        return pd.read_csv(filename)['h_' + str(layer_index) + '_' + str(neuron_index)]

