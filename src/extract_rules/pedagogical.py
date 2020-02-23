import numpy as np
import pandas as pd

from logic_manipulator.merge import merge
from rules.C5 import C5


def extract_rules(model):
    """
    Extract rules in a pedagogical manner using C5 on the outputs and inputs of the network
    """
    # Inputs to neural network. C5 requires DataFrame inputs
    X = model.get_layer_activations(layer_index=0)

    # y = output classifications of neural network. C5 requires y to be a pd.Series
    nn_model_predictions = np.argmax(model.model.predict(X), axis=1)
    y = pd.Series(nn_model_predictions)

    assert len(X) == len(y), 'Unequal number of data instances and predictions'

    # Use C5 to extract rules using only input and output values of the network
    # C5 returns disjunctive rules with conjunctive terms
    rules = C5(x=X, y=y,
               rule_conclusion_map={class_encoding.index: class_encoding.name
                                    for class_encoding in model.class_encodings},
               prior_rule_confidence=1)

    # Merge rules so that they are in Disjunctive Normal Form
    # Now there should be only 1 rule per rule conclusion
    # Ruleset is encapsulated/represented by a DNF rule
    # DNF_rules is a Dict[ClassEncoding: Rul]
    DNF_rules = merge(rules)
    assert len(DNF_rules.keys()) == len(model.class_encodings), 'Should only exist 1 DNF rule per class'

    print(DNF_rules)

    # TODO add default rules in case C5 returns no rules?
    # Return dictionary mapping from output class encoding to corresponding rule premises
    # TODO refactor this part its very inefficient right now
    class_rules = {}
    for rule in DNF_rules.values():
        print(rule)
        for output_class in model.class_encodings:
            if output_class.index == rule.get_conclusion():
                class_rules[output_class] = rule
    print(class_rules)
    # return class_rules
