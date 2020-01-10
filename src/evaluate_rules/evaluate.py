"""
Evaluate accuracy of rules extracted against training and test data
"""
from model.model import Model
from rules.term import Neuron

import numpy as np
import pandas as pd

from collections import namedtuple
NeuronData = namedtuple('NeuronData', 'neuron value')

def accuracy_on_data(model, class_rules, data_path):
    x = pd.read_csv(data_path, usecols=(lambda column: column not in ['target']))
    y = pd.read_csv(data_path, usecols=(['target']))

    neuron_vals = []
    neurons = [Neuron(layer=0, index=i) for i in range(len(x.columns))]
    for row in x.values:
        neuron_vals.append({neurons[i]: row[i] for i in range(0, len(neurons))})

    x['neuron_vals'] = neuron_vals
    y_pred = []
    for _, row in x.iterrows():
        y_pred.append(eval_row(row['neuron_vals'], class_rules, model.class_encodings))

    y['pred'] = y_pred

    y['correct'] = y['pred']==y['target']
    print('Accuracy: ', (sum(y['correct'])/len(y['correct'])))

def eval_row(neuron_vals, class_rules, output_classes):
    for output_class in output_classes:
        class_rule = class_rules[output_class]
        if (class_rule.evaluate(neuron_vals)):
            return (output_class.index)

    return np.random.randint(2)

def accuracy(model: Model):
    class_rules = model.class_rules
    accuracy_on_data(model, class_rules, model.train_data_path)

