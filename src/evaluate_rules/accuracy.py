"""
Evaluate accuracy of rules
"""

import pandas as pd

def accuracy(model):
    # Rule predictions
    y_rule_predictions = open(model.data_path + 'RULE_predictions.txt', 'r').read().split(' ')
    # Predictions saved as strings. Need to be converted to ints
    y_rule_predictions = [int(example) for example in y_rule_predictions]

    # True values
    y_correct = pd.read_csv(model.test_data_path)['target'].values

    assert (len(y_rule_predictions) == len(y_correct)), "Error: not equivalent number of values!"

    acc = sum(y_rule_predictions == y_correct) / len(y_correct)
    print('Accuracy: ', acc)
