import itertools

from rules.clause import ConjunctiveClause
from rules.rule import Rule
from rules.ruleset import Ruleset

def substitute(total_rule: Rule, intermediate_rules: Ruleset) -> Rule:
    """
    Substitute the intermediate rules from the previous layer into the total rule
    """
    new_premise_clauses = set()

    # for each clause in the total rule
    for old_premise_clause in total_rule.get_premise():
        print(old_premise_clause)
        print('.')

        # list of sets of conjunctive clauses that are all conjunctive
        conj_new_premise_clauses = []
        for old_premise_term in old_premise_clause.get_terms():
            print(old_premise_term)
            clauses_to_append = intermediate_rules.get_rule_premises_by_conclusion(old_premise_term)
            if clauses_to_append:   # TODO CHECK THIS
                conj_new_premise_clauses.append(clauses_to_append)

        # When combined into a cartesian product, get all possible conjunctive clauses for merged rule
        conj_new_premise_clauses_combinations = itertools.product(*tuple(conj_new_premise_clauses))

        # given tuples of ConjunctiveClauses
        for premise_clause_tuple in conj_new_premise_clauses_combinations:
            print('clause: ', premise_clause_tuple)
            new_clause = ConjunctiveClause()
            for premise_clause in premise_clause_tuple:
                new_clause = new_clause.union(premise_clause)

            new_premise_clauses.add(new_clause)

    return Rule(premise=new_premise_clauses, conclusion=total_rule.get_conclusion())