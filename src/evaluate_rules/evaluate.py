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
    max_class_conf = 0

    for output_class in rules.keys():
        output_class_rule = rules[output_class]

        confidence = output_class_rule.evaluate(neuron_to_value_map)

        if confidence > max_class_conf:
            max_class = output_class
            max_class_conf = confidence

    if max_class:
        return max_class.index
    else:
        return np.random.randint(2)

def accuracy(model: Model):
    data_path = model.train_data_path

    x = pd.read_csv(data_path, usecols=(lambda column: column not in ['target']))
    y = pd.read_csv(data_path, usecols=(['target']))

    # Calculate predicted class for each test_instance
    y_pred = []
    for row in x.values:
        y_pred.append(predict_test_instance(neuron_values=row, rules=model.output_class_to_rules))

    y['pred'] = y_pred
    y['correct'] = y['pred'] == y['target']

    acc = (sum(y['correct']) / len(y))
    print('Accuracy: ', acc)
    return acc




