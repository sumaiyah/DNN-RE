"""
Represent a rule with a premise in Disjunctive Normal Form (DNF) and conclusion of another term or class conclusion
"""

from rules.clause import ConjunctiveClause
from rules.term import Term, Neuron
from typing import Set, Union, Dict
from rules import DELETE_UNSATISFIABLE_CLAUSES_FLAG

from logic_manipulator.satisfiability import remove_unsatisfiable_clauses

class OutputClass:
    """
    Represent rule conclusion. Immutable and Hashable.

    Each output class has a name and its relevant encoding in the network i.e. which output neuron it corresponds to
    """
    __slots__ = ['name', 'encoding']

    def __init__(self, name: str, encoding: int):
        super(OutputClass, self).__setattr__('name', name)
        super(OutputClass, self).__setattr__('encoding', encoding)

    def __str__(self):
        return 'OUTPUT_CLASS=%s (Neuron %d)' % (self.name, self.encoding)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.name == other.name and
            self.encoding == other.encoding
        )

    # def __setattr__(self, name, value):
    #     msg = "'%s' is immutable, can't modify %s" % (self.__class__, name)
    #     raise AttributeError(msg)

    def __hash__(self):
        return hash((self.name, self.encoding))

class Rule:
    """
    Represent rule in DNF form i.e. (t1 AND t2 AND ..) OR ( ...) OR ... -> t6 . Immutable and Hashable.
    """
    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise: Set[ConjunctiveClause], conclusion: Union[Term, OutputClass]):
        if DELETE_UNSATISFIABLE_CLAUSES_FLAG:
            premise = remove_unsatisfiable_clauses(clauses=premise)

        # if DELETE_REDUNDANT_CLAUSES_FLAG:
        #     premise = self.delete_redundant_clauses(clauses=premise)

        super(Rule, self).__setattr__('premise', premise)
        super(Rule, self).__setattr__('conclusion', conclusion)

    def get_premise(self) -> Set[ConjunctiveClause]:
        return self.premise

    def get_conclusion(self) -> Union[Term, OutputClass]:
        return self.conclusion

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.premise == other.premise and
                self.conclusion == other.conclusion
        )

    def __hash__(self):
        return hash((self.conclusion))

    # def __setattr__(self, name, value):
    #     msg = "'%s' is immutable, can't modify %s" % (self.__class__, name)
    #     raise AttributeError(msg)

    def __str__(self):
        # premise_str = [(str(clause) + '\n') for clause in self.get_premise()]
        premise_str = [(str(clause)) for clause in self.premise]
        rule_str = "IF " + (' OR '.join(premise_str)) + " THEN " + str(self.conclusion)
        n_clauses = len(self.premise)

        rule_str += ('\n' + 'Number of clauses: ' + str(n_clauses))
        return rule_str

    def evaluate_rule_by_confidence(self, data: Dict[Neuron, float]) -> float:
        """
        Given a list of input neurons and their values, return the combined confidence of clauses that satisfy the rule
        """
        confidence = 0
        for clause in self.premise:
            if clause.evaluate(data):
                confidence += clause.get_confidence()

        return confidence

    def evaluate_rule_by_majority_voting(self, data: Dict[Neuron, float]) -> float:
        """
        Given a list of input neurons and their values, return the combined proportion of clauses that satisfy the rule
        """
        total = len(self.premise)
        n_satisfied_clauses = 0
        for clause in self.premise:
            if clause.evaluate(data):
                n_satisfied_clauses += 1

        return n_satisfied_clauses/total

    @classmethod
    def from_term_set(cls, premise: Set[Term], conclusion: Union[OutputClass, Term], confidence: float):
        """
        Construct Rule given a single clause as a set of terms and a colclusion
        """
        rule_premise = {ConjunctiveClause(terms=premise, confidence=confidence)}
        return cls(premise=rule_premise, conclusion=conclusion)

    @classmethod
    def initial_rule(cls, output_layer, output_class, threshold):
        """
        Construct Initial Rule given parameters with default confidence value of 1
        """
        rule_premise = ConjunctiveClause(terms={Term(Neuron(layer=output_layer,
                                                            index=output_class.encoding), '>', threshold)},
                                         confidence=1)
        rule_conclusion = output_class

        return cls(premise={rule_premise}, conclusion=rule_conclusion)

    def get_terms_with_conf_from_rule_premises(self) -> Dict[Term, float]:
        """
        Return all the terms present in the bodies of all the rules in the ruleset with their max confidence
        """
        term_confidences = {}

        for clause in self.premise:
            clause_confidence = clause.get_confidence()
            for term in clause.get_terms():
                if term in term_confidences:
                    term_confidences[term] = max(term_confidences[term], clause_confidence)
                else:
                    term_confidences[term] = clause_confidence

        return term_confidences




