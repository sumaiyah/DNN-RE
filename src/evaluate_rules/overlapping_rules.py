def overlapping_rules(rules):
    # Return the number of overlapping rules between output class rulesets
    # Want the number of clauses when intersecting the sets of clauses from all the class rule premises
    class_rule_premises = [class_rule.get_premise() for class_rule in rules]
    return  len(set.intersection(*class_rule_premises))