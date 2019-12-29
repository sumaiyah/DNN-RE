from enum import Enum
from typing import Union, Set, List

from NN_model.neural_network_model import Classifiation

class Neuron:
    """
    Represent specific neuron in the neural network
    """

    __slots__ = ['layer', 'index']

    def __init__(self, layer: int, index: int):
        self.layer = layer
        self.index = index

    def __str__(self):
        return "h_" + str(self.layer) + "," + str(self.index)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.index == other.index and
            self.layer == other.layer
        )

    def __hash__(self):
        return hash(str(self))

class TermOperator(Enum):
    GreaterThan = '>'
    LessThanEq = '<='

    def __str__(self) -> str:
        return self.value

    def inverse(self):
        """
        Return inverse of the term operator
        """
        if self.GreaterThan:
            return self.LessThanEq
        if self.LessThanEq:
            return self.GreaterThan

class Term:
    """
    Represent a term in a Rule or a condition on a branch in a RuleTree e.g. (6 > 7), (Neuron(1,2) > 0.9)
    """

    __slots__ = ['neuron', 'operator', 'operand']

    def __init__(self, neuron: Neuron, operator: str, operand):
        self.neuron = neuron
        self.operand = operand

        try:
            self.operator = TermOperator(operator)
        except ValueError as ve:
            print(ve)

    def __str__(self):
        return '(' + str(self.neuron) + ' ' + str(self.operator) + ' ' + str(self.operand) + ')'

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.neuron == other.neuron and
                self.operator == other.operator and
                self.operand == other.operand
        )

    def __hash__(self):
        return hash(str(self))

    def apply(self, value):
        if self.operator is TermOperator.GreaterThan:
            return value > self.operand

        elif self.operator is TermOperator.LessThanEq:
            return value <= self.operand

    def inverse(self):
        """
        Return inverse of the term i.e. the same term with opposite sign
        """
        return Term(self.neuron, str(self.operator.inverse()), self.operand)

    def get_neuron_index(self):
        """
        return neuron index of term neuron is applied to
        """
        return self.neuron.index

class Rule:
    """
    Represent if-then rule (path from root to leaf in RuleTree with premise of ANDed Terms and Conclusion
    """

    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise: Set[Term], conclusion: Union[Term, Classifiation]):
        self.premise = premise
        self.conclusion = conclusion
        # self.confidence = confidence TODO ADD/DO SOMETHING WITH CONDIFENCE

    def __str__(self):
        conditions_str = [str(term) for term in self.premise]
        return "IF " + (" AND ").join(conditions_str) + " THEN " + str(self.conclusion)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.premise == other.premise and
            self.conclusion == other.conclusion
        )

    def __hash__(self):
        return hash(str(self))

    def get_terms(self) -> Set[Term]:
        """
        return terms in rule premise
        """
        return self.premise

    def get_conclusion(self) -> Union[Term, Classifiation]:
        return self.conclusion

