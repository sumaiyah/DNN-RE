"""
Build neural network models given number of nodes in each hidden layer
"""
from keras import Input, Model
from keras.layers import Dense

from model.generation import DATASET_INFO

def create_model(layer_1, layer_2):
    # Input layer
    input_layer = Input(shape=(DATASET_INFO.n_inputs,))

    # Hidden layer 1
    hidden_layer_1 = Dense(layer_1, activation='tanh')(input_layer)

    # Hidden layer 2
    hidden_layer_2 = Dense(layer_2, activation='tanh')(hidden_layer_1)

    # Output Layer
    output_layer = Dense(DATASET_INFO.n_outputs, activation='softmax')(hidden_layer_2)

    # Model input-hidden-output
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile Model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model