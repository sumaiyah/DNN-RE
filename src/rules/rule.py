from enum import Enum
from typing import Union, Set


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

    __slots__ = ['operand_1', 'operator', 'operand_2']

    def __init__(self, operand_1, operator: str, operand_2):
        self.operand_1 = operand_1
        self.operand_2 = operand_2

        try:
            self.operator = TermOperator(operator)
        except ValueError as ve:
            print(ve)

    def __str__(self):
        return '(' + str(self.operand_1) + ' ' + str(self.operator) + ' ' + str(self.operand_2) + ')'

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.operand_1 == other.operand_1 and
            self.operator == other.operator and
            self.operand_2 == other.operand_2
        )

    def __hash__(self):
        return hash(str(self))

    def apply(self, value):
        if self.operator is TermOperator.GreaterThan:
            return value > self.operand_2

        elif self.operator is TermOperator.LessThanEq:
            return value <= self.operand_2

    def inverse(self):
        """
        Return inverse of the term i.e. the same term with opposite sign
        """
        return Term(self.operand_1, str(self.operator.inverse()), self.operand_2)


class ClassConclusion:
    """
    Represent conclusion of a rule assigning a classification e.g. OUT=A
    """

    __slots__ = ['class_name']

    def __init__(self, class_name: str):
        self.class_name = class_name

    def __str__(self):
        return 'class=' + self.class_name

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.class_name == other.class_name
        )

    def __hash__(self):
        return hash(str(self))

class Rule:
    """
    Represent if-then rule (path from root to leaf in RuleTree with premise of ANDed Terms and Conclusion
    """

    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise: Set[Term], conclusion: Union[Term, ClassConclusion]):
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

"""
first_tree_rules = []
first_tree_rules.append(Rule([Term(Neuron(2,1), '>', 0.6), Term(Neuron(2,4), '>', 0.3)], conclusion=ClassConclusion('0')))
first_tree_rules.append(Rule([Term(Neuron(2,1), '>', 0.6), Term(Neuron(2,4),'<=', 0.3)], conclusion=ClassConclusion('1')))
first_tree_rules.append(Rule([Term(Neuron(2,1), '<=', 0.6)], conclusion=ClassConclusion('1')))

sec_tree_rules = []
sec_tree_rules.append(Rule([Term(Neuron(1,2), '>', 0.4), Term(Neuron(1,10),'<=', 0.1)], conclusion=Term(Neuron(2,3), '<=', 0.5)))
sec_tree_rules.append(Rule([Term(Neuron(1,2), '>', 0.4), Term(Neuron(1,10),'>', 0.1)], conclusion=Term(Neuron(2,4), '>', 0.3)))

sec_tree_rules.append(Rule([Term(Neuron(1,2), '<=', 0.4), Term(Neuron(1,1),'<=', 0.4)], conclusion=Term(Neuron(2,1), '>', 0.6)))
sec_tree_rules.append(Rule([Term(Neuron(1,2), '<=', 0.4), Term(Neuron(1,1),'>', 0.4)], conclusion=Term(Neuron(2,1), '<=', 0.6)))

for rule in first_tree_rules:
    print(rule)

print()

for rule in sec_tree_rules:
    print(rule)

Substitution:
Requires taking all the terms in a rule from layer x+1
seeing if all can be matched to a conclusion of rule from layer x = rules r

if so:
    merge terms from r and keep conclusion from layer x+1
"""
