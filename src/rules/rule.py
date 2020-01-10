"""
Represent a total rule with a premise in Disjunctive Normal Form (DNF) and conclusion a class declaration
"""

from rules.clause import ConjunctiveClause
from rules.term import Term, Neuron
from typing import Set, Union, Dict
from rules import DELETE_UNSATISFIABLE_CLAUSES_FLAG, DELETE_REDUNDANT_CLAUSES_FLAG

import itertools

class Conclusion:
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

class Rule:
    """
    Represent rule in DNF form i.e. (t1 AND t2 AND ..) OR ( ...) OR ... -> t6 . Immutable and Hashable.
    """
    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise: Set[ConjunctiveClause], conclusion: Union[Term, Conclusion]):
        if DELETE_UNSATISFIABLE_CLAUSES_FLAG:
            premise = self.delete_unsatisfiable_clauses(clauses=premise)

        if DELETE_REDUNDANT_CLAUSES_FLAG:
            premise = self.delete_redundant_clauses(clauses=premise)

        super(Rule, self).__setattr__('premise', premise)
        super(Rule, self).__setattr__('conclusion', conclusion)

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

    def __setattr__(self, name, value):
        msg = "'%s' is immutable, can't modify %s" % (self.__class__,
                                            name)
        raise AttributeError(msg)

    def get_terms_from_rule_premise(self) -> Set[Term]:
        # Return terms from all clauses in rule premise with no duplicates
        terms = set()
        for clause in self.premise:
            terms = terms.union(clause.get_terms())
        return terms

    def __str__(self):
        # premise_str = [(str(clause) + '\n') for clause in self.get_premise()]
        premise_str = [(str(clause)) for clause in self.get_premise()]
        rule_str = "IF " + (' OR '.join(premise_str)) + " THEN " + str(self.get_conclusion()) + '\n'
        n_clauses = len(self.premise)
        rule_str += ('Number of clauses: ' + str(n_clauses))
        return rule_str

    @classmethod
    def from_term_set(cls, premise: Set[Term], conclusion: Union[Conclusion, Term]):
        """
        Initialise Rule given a single clause as a set of terms
        """
        rule_premise = {ConjunctiveClause(terms=premise)}
        return cls(premise=rule_premise, conclusion=conclusion)

    @classmethod
    def initial_rule(cls, output_layer, neuron_index, class_name, threshold):
        # Return initial rule given parameters
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

            # When combined into a cartesian product, get all possible conjunctive clauses for merged rule
            conj_new_premise_clauses_combinations = itertools.product(*tuple(conj_new_premise_clauses))

            # given tuples of ConjunctiveClauses
            for premise_clause_tuple in conj_new_premise_clauses_combinations:
                new_clause = ConjunctiveClause()
                for premise_clause in premise_clause_tuple:
                    new_clause = new_clause.union(premise_clause)

                new_premise_clauses.add(new_clause)

        return Rule(premise=new_premise_clauses, conclusion=self.get_conclusion())

    def delete_unsatisfiable_clauses(self, clauses: Set[ConjunctiveClause]) -> Set[ConjunctiveClause]:
        """
        Remove unsatisfiable clauses in a rule
        """
        satisfiable_clauses = set()
        for clause in clauses:
            if clause.is_satisfiable():
                satisfiable_clauses.add(clause)

        return satisfiable_clauses

    def delete_redundant_clauses(self, clauses: Set[ConjunctiveClause]) -> Set[ConjunctiveClause]:
        """
        TODO FILL THIS IN!!
        Remove redundant clauses in a rule

        A clause c1 is strictly weaker than another clause c2 in the rule
            - if it includes terms for all dimensions delimited by r2
            - the restrictions are at least as specific as those of r2 (r2 is more specifc i.e. less general)
        """
        return clauses

    def evaluate(self, data: Dict[Neuron, float]) -> bool: #todo replace data with dataframe
        """
        Given a list of input neurons and their values, return whether this rule applied to the data can be satisfied
        """
        # If at least 1 clause is satisfied
        for clause in self.premise:
            if (clause.evaluate(data)):
                return True

        # Could not satisfy any of the clauses
        return False

# c = ConjunctiveClause({Term(Neuron(0,1),'>', 0.5), Term(Neuron(0,2),'>', 0.5),})
# r = Rule({c}, Conclusion('hi'))
# for x in r.premise:
#     x.terms = {}
# print(r)
# data = {Neuron(0,1): 0.74, Neuron(0,2): 0.5}
# print(r.apply(data))