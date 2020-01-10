from typing import Set

from rules.term import Term, TermOperator
from rules import DELETE_REDUNDANT_TERMS_FLAG

class ConjunctiveClause:
    """
    Represent conjunctive clause. All terms in clause are ANDed together. Immutable and Hashable.
    """
    __slots__ = ['terms']

    def __init__(self, terms: Set[Term] = None):
        if terms is None:
            terms = set()

        if DELETE_REDUNDANT_TERMS_FLAG:
            terms = self.__get_necessary_terms(terms)

        super(ConjunctiveClause, self).__setattr__('terms', terms)

    def __str__(self):
        terms_str = [str(term) for term in self.terms]
        return '[' + ' AND '.join(terms_str) + ']'

    def __setattr__(self, name, value):
        msg = "'%s' is immutable, can't modify %s" % (self.__class__,
                                            name)
        raise AttributeError(msg)

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.terms == other.terms
        )

    def __hash__(self):
        x = hash(1)
        for term in self.terms:
            x = x ^ hash(term)
        return x

    def get_terms(self) -> Set[Term]:
        return self.terms

    def union(self, other) -> 'ConjunctiveClause':
        # Return new conjunctive clause that has all terms from both
        terms = self.get_terms().union(other.get_terms())
        return ConjunctiveClause(terms)

    def __get_necessary_terms(self, terms) -> Set[Term]:
        """
        Remove redundant terms from a clause
        """
        neuron_conditions = get_neuron_dict(terms) # {Neuron: {TermOperator: [Float]}}
        necessary_terms = set()

        # Find most general neuron thresholds, range as general as possible, for '>' keep min, for '<=' keep max
        for neuron in neuron_conditions.keys():
            for TermOp in TermOperator:
                if neuron_conditions[neuron][TermOp]:  # if non-empty list
                    necessary_terms.add(
                        Term(neuron, TermOp, TermOp.most_general_value(neuron_conditions[neuron][TermOp])))

        return necessary_terms

    def is_satisfiable(self) -> bool:
        """
        Return whether or not the clause is satisfiable. Unsatisfiable if a neurons min value > its max value
        """
        neuron_conditions = get_neuron_dict(self.terms)  # {Neuron: {TermOperator: Float or []}}

        for neuron in neuron_conditions.keys():
            min_value = neuron_conditions[neuron][TermOperator.GreaterThan]
            max_value = neuron_conditions[neuron][TermOperator.LessThanEq]

            if min_value and max_value:
                if min_value > max_value:
                    return False

        return True

    def evaluate(self, data) -> bool:
        """
        Evaluate clause with data Dict[Neuron, float]
        """
        # All terms in the clause must be true
        for term in self.get_terms():
            if not term.apply(data[term.get_neuron()]):
                return False

        return True

def get_neuron_dict(terms: Set[Term]):
    # Return {Neuron: {TermOperator: [Float]}}
    neuron_conditions = {}

    for term in terms:
        if not term.get_neuron() in neuron_conditions:  # unseen Neuron
            neuron_conditions[term.get_neuron()] = {TermOp: [] for TermOp in TermOperator}
        neuron_conditions[(term.get_neuron())][term.get_operator()].append(term.get_threshold())

    return neuron_conditions