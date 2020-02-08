"""
Represent trained Neural Network model
"""

import pandas as pd
import keras.models as keras


class Model:
    """
    Represent trained neural network model
    """

    def __init__(self, class_encodings, data_path, recompute_layer_activations):
        model_path = data_path + 'model.h5'
        self.model: keras.Model = keras.load_model(model_path)
        self.data_path = data_path
        self.class_encodings = class_encodings

        self.output_class_to_rules = {}  # DNF rule for each output class
        self.n_layers = len(self.model.layers)

        self.train_data_path = data_path + 'train_data.csv'
        self.test_data_path = data_path + 'test_data.csv'

        if recompute_layer_activations:
            self.__compute_layerwise_activations()

    def __compute_layerwise_activations(self):
        """
        Store sampled activations for each layer in CSV files
        """
        # todo make this method work for func and non func keras models

        # Only need input values
        data_x = pd.read_csv(self.train_data_path).drop(columns=['target'])

        # Sample network at each layer
        for layer_index in range(0, self.n_layers):
            partial_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)

            # e.g. h_1_0, h_1_1, ...
            neuron_labels = ['h_' + str(layer_index) + '_' + str(i)
                             for i in range(0, self.model.layers[layer_index].output_shape[1])]

            activation_values = pd.DataFrame(data=partial_model.predict(data_x), columns=neuron_labels)
            activation_values.to_csv(self.data_path + str(layer_index) + '.csv', index=False)

    def get_layer_activations(self, layer_index: int):
        """
        Return activation values given layer index
        """
        filename = self.data_path + str(layer_index) + '.csv'
        return pd.read_csv(filename)

    def get_layer_activations_of_neuron(self, layer_index: int, neuron_index: int):
        """
        Return activation values given layer index, only return the column for a given neuron index
        """
        filename = self.data_path + str(layer_index) + '.csv'
        return pd.read_csv(filename)['h_' + str(layer_index) + '_' + str(neuron_index)]

    def save_rules(self):
        import pickle
        print('Writing rules to disk...', end='', flush=True)
        with open('extracted_rules.txt', 'w') as output:
            for rule in self.output_class_to_rules.values():
                output.write(str(rule) + '\n')
        print('done')

    def set_rules(self, rules):
        self.output_class_to_rules = rules

    def print_rules(self):
        for output_class in self.output_class_to_rules.keys():
            print('CLASS: ', output_class.name)
            print(self.output_class_to_rules[output_class])
            print()
