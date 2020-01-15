from typing import Set

from rules.clause import ConjunctiveClause
from logic_manipulator.helpers import terms_set_to_neuron_dict
from rules.term import TermOperator

def is_satisfiable(clause: ConjunctiveClause):
    """
    Return whether or not the clause is satisfiable. Unsatisfiable if a neurons min value > its max value
    """
    neuron_conditions = terms_set_to_neuron_dict(clause.get_terms())

    for neuron in neuron_conditions.keys():
        # If neuron is specified with <= and >
        if neuron_conditions[neuron][TermOperator.GreaterThan] and neuron_conditions[neuron][TermOperator.LessThanEq]:
            gt_vals = neuron_conditions[neuron][TermOperator.GreaterThan]
            lteq_vals = neuron_conditions[neuron][TermOperator.LessThanEq]


            if gt_vals and lteq_vals: # if neuron is subject to both predicates
                min_value = min(gt_vals)
                max_value = max(lteq_vals)
                if min_value >= max_value:
                    return False

    # All conditions on a neuron are satisfiable
    return True

def remove_unsatisfiable_clauses(clauses: Set[ConjunctiveClause]):
    satisfiable_clauses = set()
    """
    Remove unsatisfiable clauses 
    """
    for clause in clauses:
        if is_satisfiable(clause):
            satisfiable_clauses.add(clause)

    return satisfiable_clauses