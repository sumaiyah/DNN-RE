from evaluate_rules.predict import predict
from evaluate_rules.accuracy import accuracy
from evaluate_rules.fidelity import fidelity
from evaluate_rules.comprehensibility import comprehensibility

def evaluate(model):
    """
    Evaluate ruleset generated
    """
    # Use ruleset to classify test data
    predict(model)

    # Compute Accuracy
    accuracy(model)

    # Compute Fidelity
    fidelity(model)

    # Compute Comprehensibility
    comprehensibility(model)

    # Compare features used
    # # TODO refactor this into its own .py file
    # features = set()
    # for class_rule in model.output_class_to_rules.values():
    #     for clause in class_rule.get_premise():
    #         for term in clause.get_terms():
    #             features.add(term.get_neuron())
    # return features
