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

def predict_test_instance(input, rules):
    input_neuron_values = {Neuron(layer=0, index=i): input[i] for i in range(len(input))}

    max_class = None
    max_class_conf = 0

    for output_class in rules.keys():
        output_class_dnf_formula = rules[output_class]

        confidence = output_class_dnf_formula.evaluate(input_neuron_values)

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
        y_pred.append(predict_test_instance(input=row ,rules=model.output_class_to_dnf_formula))

    y['pred'] = y_pred
    y['correct'] = y['pred'] == y['target']
    print('Accuracy: ', (sum(y['correct']) / len(y)))

# to remove overlapping clauses
# rule_premises = [rule.get_premise() for rule in model.output_class_to_dnf_formula.values()]
# overlapping_clauses = set.intersection(*rule_premises)
