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