"""
Compute comprehensibility of ruleset generated

- Number of rules per class = number of conjunctive clauses in a classes DNF
- Number of terms per rule: Min, Max, Average
"""

def comprehensibility(rules):
    for rule in rules:
        class_name = rule.get_conclusion().name

        # Number of rules in that class
        n_rules_in_class = len(rule.get_premise())

        #  Get min max average number of terms in a clause
        min_n_terms = float('inf')
        max_n_terms = 0
        total_n_terms = 0
        for clause in rule.get_premise():
            # Number of terms in the clause
            n_clause_terms = len(clause.get_terms())

            if n_clause_terms < min_n_terms:
                min_n_terms = n_clause_terms
            if n_clause_terms > max_n_terms:
                max_n_terms = n_clause_terms

            total_n_terms += n_clause_terms

        av_n_terms_per_rule = total_n_terms / n_rules_in_class

        print('class:%s n_rules:%d min_n_terms:%d max_n_terms:%d av_n_terms_per_rule:%f' %
         (class_name, n_rules_in_class, min_n_terms, max_n_terms, av_n_terms_per_rule))

        # For each class: n_rules_in_class, min_n_terms, max_n_terms, av_n_terms_per_rule
        yield class_name, n_rules_in_class, min_n_terms, max_n_terms, av_n_terms_per_rule
