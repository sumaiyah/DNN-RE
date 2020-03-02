"""
Uses extracted rules to classify new instances from test data and store results in file
"""
import random
import numpy as np
import pandas as pd
import operator


from rules.term import Neuron

# TODO this code is highly parallelizable!
def classify_instance(model, inputs):
    # 1 DNF rule per class. Rule conclusion is the output class
    rules = model.rules

    # Map of Neuron objects to values from input data
    neuron_to_value_map = {Neuron(layer=0, index=i): inputs[i] for i in range(len(inputs))}

    # Try each rule with input data and get a score
    rule_scores = {}
    for rule in rules:
        # Score indicates how confident the rule is classifying this instance as the class
        score = rule.evaluate_rule_by_majority_voting(neuron_to_value_map)
        # score = rule.evaluate_rule_by_confidence(neuron_to_value_map)
        rule_scores[rule] = score

    # Assign score to each output class.
    # Class with max score decides the classification of the instance
    # If tie, choose randomly
    if len(set(rule_scores.values())) == 1:
        max_class = random.choice(list(rules)).conclusion
    else:
        max_class = max(rule_scores, key=rule_scores.get).conclusion

    # Return encoding of max rule i.e. neuron value
    return max_class.encoding

def predict(model, data):
    """
    Use rules to make predictions about test data and return those predictions
    """
    # Evaluate each test instance separately
    y_pred = []
    for data_instance_x in data:
        y_pred.append(classify_instance(model=model, inputs=data_instance_x))

    return y_pred

