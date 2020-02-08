"""
Compute comprehensibility of ruleset generated

- Number of rules per class = number of conjunctive clauses in a classes DNF
- Number of terms per rule: Min, Max, Average
"""

def comprehensibility(model):
    print('Comprehensibility:  ')
    rules = model.output_class_to_rules

    # Total number of terms in all rules
    overall_total = 0

    for output_class in rules.keys():
        print('    Class: ', output_class.name, end=' | ')
        class_rule = rules[output_class]

        # Number of rules per class
        n_rules = len(class_rule.get_premise())
        print('n_rules: %d' % n_rules, end=' | ')

        # Get min max average number of terms in a clause
        min_n_terms = float('inf')
        max_n_terms = class_total = 0
        for clause in class_rule.get_premise():
            n_terms = len(clause.get_terms())
            if n_terms < min_n_terms:
                min_n_terms = n_terms
            if n_terms > max_n_terms:
                max_n_terms = n_terms

            class_total += n_terms
            overall_total += n_terms

        # Average number of terms in a rule
        average_in_class = class_total / n_rules

        print('number of terms: min: %d max: %d average: %f | ' % (min_n_terms, max_n_terms, average_in_class))

    # Average number of terms in a rule over entire ruleset
    total_n_rules = sum([len(rules[output_class].get_premise()) for output_class in rules.keys()])
    average_overall = overall_total / total_n_rules
    print('  Overall average n_terms per rule: ', average_overall)