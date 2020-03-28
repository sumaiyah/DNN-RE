def overlapping_features(rules, include_operand=False):
    # Return the number of overlapping features considered in output class rulesets
    # TODO: If include operand: consider feature as a threshold on an input feature
    # TODO:this would require comparing 2 thresholds if they have the same sign but the value of threshold can differ

    all_features = []
    for class_rule in rules:
        class_features = set()
        for clause in class_rule.get_premise():
            for term in clause.get_terms():
                class_features.add(term.get_neuron())
        all_features.append(class_features)

    # Intersection over features used in each rule
    return len(set.intersection(*all_features))
