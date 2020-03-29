"""
Find best neural network initialisation

1. Load train test split
2. Build 5 Neural Networks with different initialisations using besy hyper parameters
3. Perform rule extraction on these 5 networks
4. network with smallest ruleset, save that initialisation
"""
import os

import numpy as np
import pandas as pd

from src import INITIALISATIONS_DIR, TEMP_DIR, RESULTS_DIR, RULE_EX_MODE
from model.generation.helpers import split_data
from model.generation.helpers.build_and_train_model import build_and_train_model
import dnn_re
from keras.models import load_model


def run(X, y, hyperparameters):
    train_index, test_index = split_data.load_split_indices(file_path=INITIALISATIONS_DIR + 'data_split_indices.txt')

    # Split data
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    # Save information about nn initialisation
    results_file_path = RESULTS_DIR + 'nn_initialisations.csv'
    if not os.path.exists(results_file_path):
        pd.DataFrame(data=[],
                     columns=['run',
                              'nn_hyperparams',
                              'nn_acc',
                              're_acc',
                              're_fid',
                              're_time',
                              're_memory',
                              're_n_rules']).to_csv(
            results_file_path, index=False)

    # Path to trained neural network
    model_file_path = TEMP_DIR + 'model.h5'

    # Smallest ruleset i.e. total number of rules
    smallest_ruleset_size = np.float('inf')
    smallest_ruleset_acc = 0

    label_file_path = TEMP_DIR + 'labels.csv'

    for i in range(0, 5):
        print('Testing initialisation %d' % i)

        # Build and train nn put it in temp/
        nn_accuracy = build_and_train_model(X_train, y_train, X_test, y_test, **hyperparameters, model_file_path=model_file_path)

        # Extract rules
        re_results = dnn_re.run(X, y, train_index, test_index, model_file_path, label_file_path)

        # Save rule extraction results results
        results_df = pd.read_csv(results_file_path)
        results_df = results_df.append({
            'run': i,
            'nn_hyperparams': hyperparameters,
            'nn_acc': nn_accuracy,
            're_mode': RULE_EX_MODE.mode,
            're_acc': re_results['acc'],
            're_fid': re_results['fid'],
            're_time': re_results['time'],
            're_memory': re_results['memory'],
            're_n_rules_per_class': re_results['n_rules_per_class']}, ignore_index=True)
        results_df.to_csv(results_file_path, index=False)

        # If this initialisation extrcts a smaller ruleset - save it
        ruleset_size = sum(re_results['n_rules_per_class'])
        if (ruleset_size < smallest_ruleset_size) \
                or (ruleset_size == smallest_ruleset_size and re_results['acc'] > smallest_ruleset_acc):
            # Save initilisation as best_initialisation.h5
            load_model(TEMP_DIR + 'initialisation.h5').save(INITIALISATIONS_DIR + 'best_initialisation.h5')

    print('Found neural network with the best initialisation.')
