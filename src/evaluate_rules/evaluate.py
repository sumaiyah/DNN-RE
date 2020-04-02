from collections import OrderedDict

import pandas as pd

from evaluate_rules.overlapping_features import overlapping_features
from evaluate_rules.overlapping_rules import overlapping_rules
from evaluate_rules.predict import predict
from evaluate_rules.accuracy import accuracy
from evaluate_rules.fidelity import fidelity
from evaluate_rules.comprehensibility import comprehensibility
import os

from src import RULE_EXTRACTOR


def evaluate(rules, label_file_path):
    """
    Evaluate ruleset generated
    """
    labels_df = pd.read_csv(label_file_path)

    predicted_labels = labels_df['rule_%s_labels' % RULE_EXTRACTOR.mode]
    true_labels = labels_df['true_labels']
    nn_labels = labels_df['nn_labels']

    # Compute Accuracy
    acc = accuracy(predicted_labels, true_labels)

    # Compute Fidelity
    fid = fidelity(predicted_labels, nn_labels)

    # Compute Comprehensibility
    comprehensibility_results = comprehensibility(rules)

    n_overlapping_features = overlapping_features(rules)

    results = OrderedDict(acc=acc, fid=fid, n_overlapping_features=n_overlapping_features)
    results.update(comprehensibility_results)

    return results

