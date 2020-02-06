"""
Evaluate accuracy of rules extracted against training and test data
"""
from model.model import Model
from rules.rule import Rule
from rules.term import Neuron

import numpy as np
import pandas as pd

from collections import namedtuple
NeuronData = namedtuple('NeuronData', 'neuron value')

def predict_test_instance(neuron_values, rules):
    neuron_to_value_map = {Neuron(layer=0, index=i): neuron_values[i] for i in range(len(neuron_values))}

    max_class = None
    max_class_score = 0

    for output_class in rules.keys():
        output_class_rule = rules[output_class]

        # score = output_class_rule.evaluate_rule_by_confidence(neuron_to_value_map)
        score = output_class_rule.evaluate_rule_by_majority_voting(neuron_to_value_map)

        if score > max_class_score:
            max_class = output_class
            max_class_score = score

    if max_class:
        return max_class.index
    else:
        return np.random.randint(2)

def accuracy(model: Model):
    print('Calculating Accuracy\n')
    data_path = model.train_data_path

    x = pd.read_csv(data_path, usecols=(lambda column: column not in ['target']))
    y = pd.read_csv(data_path, usecols=(['target']))

    # Progress bar
    i = 0
    for row in x.values:
        if i % 1000 == 0:
            print('.', end='')
        i += 1
    print()
    i = 0
    # Calculate predicted class for each test_instance
    y_pred = []
    for row in x.values:
        y_pred.append(predict_test_instance(neuron_values=row, rules=model.output_class_to_rules))

        # Progress bar
        if i % 1000 == 0:
            print('.', end='', flush=True)
        i += 1
    print()

    y['pred'] = y_pred
    y['correct'] = y['pred'] == y['target']

    acc = (sum(y['correct']) / len(y))
    print('Accuracy: ', acc)
    return acc




