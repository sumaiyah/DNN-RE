"""
Build neural network models given number of nodes in each hidden layer
"""
from keras import Input, Model
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.utils import class_weight
import numpy as np

from model.generation import DATASET_INFO, TEMP_DIR, INITIALISATIONS_DIR
from keras.models import load_model

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


def build_and_train_model(X_train, y_train, X_test, y_test, batch_size, epochs, layer_1, layer_2, model_file_path,
                          with_best_initilisation_flag=False):
    """

    Args:
        X_train:
        y_train:
        X_test:
        y_test:
        batch_size:
        epochs:
        layer_1:
        layer_2:
        model_file_path: path to store trained nn model
        with_best_initilisation_flag: if true, use initialisation saved as best_initialisation.h5

    Returns:
        model_accuracy: accuracy of nn model
        nn_predictions: predictions made by nn used for rule extraction
    """

    # To get 2 node output make y categorical
    y_train_cat, y_test_cat = to_categorical(y_train), to_categorical(y_test)

    # Weight classes due to imbalanced dataset
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

    if with_best_initilisation_flag:
        # Use best saved initialisation found earlier
        best_initialisation_file_path = INITIALISATIONS_DIR + 'best_initialisation.h5'
        model = load_model(best_initialisation_file_path)
    else:
        # Build and initialise new model
        model = create_model(layer_1, layer_2)
        model.save(TEMP_DIR + 'initialisation.h5')

    # Train Model
    model.fit(X_train,
              y_train_cat,
              class_weight=class_weights,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    # Evaluate Accuracy of the Model
    _, nn_accuracy = model.evaluate(X_test, y_test_cat)
    print('model acc %f' % (nn_accuracy))

    # Save Trained Model
    model.save(model_file_path)

    return nn_accuracy
