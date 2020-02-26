from evaluate_rules.predict import predict
from evaluate_rules.accuracy import accuracy
from evaluate_rules.fidelity import fidelity
from evaluate_rules.comprehensibility import comprehensibility

def evaluate(model, predicted_labels, network_labels, true_labels, runtime_data_path):
    """
    Evaluate ruleset generated
    """
    # Compute Accuracy
    acc = accuracy(predicted_labels, true_labels)

    # Compute Fidelity
    fid = fidelity(predicted_labels, network_labels)

    # Save Evaluation metrics to disk
    with open(runtime_data_path, 'a') as file:
        file.write('Acc: %f ' % acc)
        file.write('Fid: %f \n' % fid)

        # Compute Comprehensibility (Number of rules per class and length of the rules)
        for class_name, n_rules_in_class, min_n_terms, max_n_terms, av_n_terms_per_rule in comprehensibility(model.rules):
            file.write('class:%s n_rules:%d min_n_terms:%d max_n_terms:%d av_n_terms_per_rule:%f \n' %
                       (class_name, n_rules_in_class, min_n_terms, max_n_terms, av_n_terms_per_rule))

    # Compare features used
    # # TODO refactor this into its own .py file
    # features = set()
    # for class_rule in model.output_class_to_rules.values():
    #     for clause in class_rule.get_premise():
    #         for term in clause.get_terms():
    #             features.add(term.get_neuron())
    # return features

