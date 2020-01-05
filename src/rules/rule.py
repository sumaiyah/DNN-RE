"""
Represent a total rule with a premise in Disjunctive Normal Form (DNF) and conclusion a class declaration
"""

from rules.term import Term, Neuron
# from rules.ruleset import Ruleset
from typing import Set, Union

import itertools

class Conclusion:
    # todo does this even need to be hashable/immutble??
    """
    Represent rule conclusion. Immutable and Hashable.
    """
    __slots__ = ['class_name']

    def __init__(self, class_name: str):
        super(Conclusion, self).__setattr__('class_name', class_name)

    def __str__(self):
        return 'OUT=' + self.class_name

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.class_name == other.class_name
        )

    def __setattr__(self, name, value):
        msg = "'%s' is immutable, can't modify %s" % (self.__class__,
                                            name)
        raise AttributeError(msg)

    def __hash__(self):
        return hash((self.class_name))

class ConjunctiveClause:
    """
    Represent conjunctive clause. All terms in clause are ANDed together. Immutable and Hashable.
    """
    __slots__ = ['terms']

    def __init__(self, terms: Set[Term] = None):
        if terms is None:
            terms = set()
        self.terms = terms

    def __str__(self):
        terms_str = [str(term) for term in self.terms]
        return '[' + ' AND '.join(terms_str) + ']'

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
        # return new conjunctive clause that has all terms from both
        terms = self.get_terms().union(other.get_terms())
        return ConjunctiveClause(terms)

class Rule:
    """
    Represent rule in DNF form i.e. (t1 AND t2 AND ..) OR ( ...) OR ... -> t6 _
    """
    def __init__(self, premise: Set[ConjunctiveClause], conclusion: Union[Term, Conclusion]):
        self.premise = premise
        self.conclusion = conclusion

    def get_premise(self) -> Set[ConjunctiveClause]:
        return self.premise

    def get_conclusion(self) -> Union[Term, Conclusion]:
        return self.conclusion

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.premise == other.premise and
                self.conclusion == other.conclusion
        )

    def __hash__(self):
        return hash((self.conclusion))

    def get_terms_from_rule_premise(self) -> Set[Term]:
        # return terms from all premises with no duplicates
        terms = set()
        for clause in self.premise:
            terms = terms.union(clause.get_terms())
        return terms

    def __str__(self):
        premise_str = [str(clause) for clause in self.get_premise()]
        return "IF " + (' OR '.join(premise_str)) + " THEN " + str(self.get_conclusion())

    @classmethod
    def from_term_set(cls, premise: Set[Term], conclusion: Union[Conclusion, Term]):
        """
        Initialise Rule given a single clause as a set of terms
        """
        rule_premise = {ConjunctiveClause(terms=premise)}
        return cls(premise=rule_premise, conclusion=conclusion)

    @classmethod
    def initial_rule(cls, output_layer, neuron_index, class_name, threshold):
        rule_premise = ConjunctiveClause(terms={Term(Neuron(layer=output_layer, index=neuron_index), '>', threshold)})
        rule_conclusion = Conclusion(class_name)

        return cls(premise={rule_premise}, conclusion=rule_conclusion)

    def merge(self, intermediate_rules: 'Ruleset') -> 'Rule':
        """
        Merge the total rule with the set of intermediate rules from the previous layer
        """
        new_premise_clauses = set()

        # for each clause in the total rule
        for old_premise_clause in self.get_premise():

            # list of sets of conjunctive clauses that are all conjunctive
            conj_new_premise_clauses = []
            for old_premise_term in old_premise_clause.get_terms():
                conj_new_premise_clauses.append(intermediate_rules.get_rule_premises_by_conclusion(old_premise_term))

            # When combined into a cartesian product, get all possible concjunctive clauses for merged rule
            conj_new_premise_clauses_combinations = itertools.product(*tuple(conj_new_premise_clauses))

            # given tuples of ConjunctiveClauses
            for premise_clause_tuple in conj_new_premise_clauses_combinations:
                new_clause = ConjunctiveClause()
                for premise_clause in premise_clause_tuple:
                    new_clause=new_clause.union(premise_clause)

                new_premise_clauses.add(new_clause)

        return Rule(premise=new_premise_clauses, conclusion=self.get_conclusion())



"""
- write tests for merging on simple data

- delete unsatisfiable rules
- delete redundant rules

- get both versions of the alg working and compare 
    - have comparison metrics for the alg set up


"""
