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

        score = class_rule.evaluate_rule_by_confidence(neuron_to_value_map)
        if score > max_class_score:
            max_class = output_class
            max_class_score = score

    # If tie between 2 classes, return random class
    if max_class:
        return max_class.index
    else:
        return np.random.randint(len(rules.keys()))

def predict(model):
    """
    Use rules to make predictions about test data
    """
    # Load test data
    X_test = pd.read_csv(model.test_data_path).drop(columns=['target'])

    # Evaluate each test instance separately
    y_pred = []
    for instance in X_test.values:
        y_pred.append(classify_instance(model=model, inputs=instance))

    # Save rule predictions
    with open(model.data_path + 'RULE_predictions.txt', 'w') as file:
        file.write(' '.join([str(pred) for pred in y_pred]))
