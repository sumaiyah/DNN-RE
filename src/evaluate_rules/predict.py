"""
Uses extracted rules to classify new instances from test data and store results in file
"""
import numpy as np
import pandas as pd

from rules.term import Neuron

def classify_instance(model, inputs):
    # TODO this code is highly parallelizable!
    # Rules used to classify instance
    rules = model.output_class_to_rules

    # Map of Neuron instances to values
    neuron_to_value_map = {Neuron(layer=0, index=i): inputs[i] for i in range(len(inputs))}

    # Assign score to each output class. Class with max score decides the classification of the instance
    max_class = None
    max_class_score = 0
    for output_class in rules.keys():
        # 1 DNF rule per classification
        class_rule = rules[output_class]

        score = class_rule.evaluate_rule_by_majority_voting(neuron_to_value_map)
        if score > max_class_score:
            max_class = output_class
            max_class_score = score

    # If tie between 2 classes, return random class
    if max_class:
        return max_class.index
    else:
        return np.random.randint(len(rules.keys()))

def predict(model, data):
    """
    Use rules to make predictions about test data and return those predictions
    """
    # Evaluate each test instance separately
    y_pred = []
    for data_instance_x in data:
        y_pred.append(classify_instance(model=model, inputs=data_instance_x))

    return y_pred

