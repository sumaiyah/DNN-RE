from enum import Enum
from typing import Union, Set, List

from logic.clause import DisjunctiveClause, ConjunctiveClause, Clause
from logic.normal_form import DNF
from logic.literal import Literal


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

    def get_index(self):
        return self.index

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

class Term(Literal):
    """
    Represent a term in a Rule or a condition on a branch in a RuleTree e.g. (6 > 7), (Neuron(1,2) > 0.9)
    """

    __slots__ = ['neuron', 'operator', 'operand']

    def __init__(self, neuron: Neuron, operator: str, operand: float):
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
        return hash((self.neuron, self.operator, self.operand))

    def apply(self, value):
        if self.operator is TermOperator.GreaterThan:
            return value > self.operand

        elif self.operator is TermOperator.LessThanEq:
            return value <= self.operand

    def negate(self):
        """
        Return inverse of the term i.e. the same term with opposite sign
        """
        return Term(self.neuron, str(self.operator.inverse()), self.operand)

    def get_neuron_index(self):
        """
        return neuron index of term neuron is applied to
        """
        return self.neuron.get_index()

class Classifiation(Literal):
    """
    Represent classification of instance with index used when encoding instanc in network
    """

    __slots__ = ['class_name', 'class_index']

    def __init__(self, class_name: str, class_index: int):
        self.class_name = class_name
        self.class_index = class_index

    def __str__(self):
        return 'OUT=' + self.class_name

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.class_name == other.class_name
        )

    def __hash__(self):
        return hash(str(self))

    def negate(self):
        print('implement this!')

class Rule:
    """
    Represent if-then rule as an object with DNF premise and Term/Classification conclusion
    """

    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise: DNF(clauses=Set[ConjunctiveClause]), conclusion: Union[Term, Classifiation]):
        self.premise = premise
        self.conclusion = conclusion
        # self.confidence = confidence TODO ADD/DO SOMETHING WITH CONDIFENCE

    def __str__(self):
        return "IF " + str(self.premise) + " THEN " + str(self.conclusion)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.premise == other.premise and
            self.conclusion == other.conclusion
        )

    def __hash__(self):
        return hash(str(self))

    def get_premise_clauses(self) -> Set[Clause]:
        return self.premise.get_clauses()

    def get_premise(self) -> DNF:
        return self.premise

    def get_conclusion(self) -> Union[Term, Classifiation]:
        return self.conclusion
