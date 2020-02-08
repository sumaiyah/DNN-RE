"""
Evaluate fidelity of rules generated i.e. how well do they mimic the performance of the Neural Network
"""

import pandas as pd

def fidelity(model):
    # Rule predictions
    y_rule_predictions = open(model.data_path + 'RULE_predictions.txt', 'r').read().split(' ')

    # Neural Network predictions
    y_nn_predictions = open(model.data_path + 'nn_predictions.txt', 'r').read().split(' ')

    assert (len(y_rule_predictions) == len(y_nn_predictions)), "Error: not equivalent number of values!"

    # Finds matching elements between the 2 lists
    matches = [y_rule_predictions[i]==y_nn_predictions[i] for i in range(len(y_rule_predictions))]

    acc = sum(matches) / len(y_nn_predictions)
    print('Fidelity: ', acc)
