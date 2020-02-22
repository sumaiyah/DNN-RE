import pandas as pd

from logic_manipulator.merge import merge
from rules.C5 import C5


def extract_rules(model):
    """
    Extract rules in a pedagogical manner using C5 on the outputs and inputs of the network
    """
    data = pd.read_csv(model.train_data_path)

    # x = train data input features
    # TODO change this so it works with the original data? Can data cope and return rules where x parameters are
    #  replaced with input feature real names
    # x = data.drop(['target'], axis=1)
    x = model.get_layer_activations(0)

    # y = train data data target variable
    y = data['target']

    # Use C5 to get rules for the output class based on the input values
    rules = C5(x=x, y=y,
               rule_conclusion_map={0: 0, 1: 1}, # for binary classification
               # rule_conclusion_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
               prior_rule_confidence=1)

    # print(rules)

    # Merge rules so that they are in Disjunctive Normal Form
    DNF_rules = merge(rules)
    # Should only exist 1 DNF rule per class
    # assert len(DNF_rules) >= len(model.class_encodings), "Error: Fewer rules than classes generated for C5"
    # TODO add default rules in case C5 returns no rules?

    # Return dictionary mapping from output class encoding to corresponding rule premises
    # TODO refactor this part its very inefficient right now
    class_rules = {}
    for rule in DNF_rules:
        for output_class in model.class_encodings:
            if output_class.index == rule.get_conclusion():
                class_rules[output_class] = rule
    return class_rules
