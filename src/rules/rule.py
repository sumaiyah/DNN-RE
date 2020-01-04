"""
Represent rules - all classes here are immutable and hashable
"""

from enum import Enum
from typing import Set, Union

class TermOperator(Enum):
    GreaterThan = '>'
    LessThanEq = '<='

    def __str__(self) -> str:
        return self.value

    def negate(self):
        if self.GreaterThan:
            return self.LessThanEq
        if self.LessThanEq:
            return self.GreaterThan

    def eval(self):
        import operator
        if self.GreaterThan:
            return operator.gt
        if self.LessThanEq:
            return operator.le

class Neuron:
    """
    Represent specific neuron in the neural network. Immutable and Hashable.
    """

    __slots__ = ['layer', 'index']

    def __init__(self, layer: int, index: int):
        super(Neuron, self).__setattr__('layer', layer)
        super(Neuron, self).__setattr__('index', index)

    def __str__(self):
        return 'h_' + str(self.layer) + ',' + str(self.index)

    def __setattr__(self, name, value):
        msg = "'%s' is immutable, can't modify %s" % (self.__class__,
                                            name)
        raise AttributeError(msg)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.index == other.index and
            self.layer == other.layer
        )

    def __hash__(self):
        return hash((self.layer, self.index))

    def get_index(self):
        return self.index

class Term:
    """
    Represent a condition indicating if activation value of neuron is above/below a threshold. Immutable and Hashable.
    """

    __slots__ = ['neuron', 'operator', 'threshold']

    def __init__(self, neuron: Neuron, operator: str, threshold: float):
        super(Term, self).__setattr__('neuron', neuron)
        super(Term, self).__setattr__('threshold', threshold)

        operator: TermOperator = TermOperator(operator)
        super(Term, self).__setattr__('operator', operator)

    def __str__(self):
        return '(' + str(self.neuron) + ' ' + str(self.operator) + ' ' + str(self.threshold) + ')'

    def __setattr__(self, name, value):
        msg = "'%s' is immutable, can't modify %s" % (self.__class__,
                                            name)
        raise AttributeError(msg)

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.neuron == other.neuron and
                self.operator == other.operator and
                self.threshold == other.threshold
        )

    def __hash__(self):
        return hash((self.neuron, self.operator, self.threshold))

    def negate(self) -> 'Term':
        """
        Return term with opposite sign
        """
        return Term(self.neuron, str(self.operator.negate()), self.threshold)

    def apply(self, value):
        """
        Apply condition to a value
        """
        return self.operator.eval()(value, self.threshold)

    def get_neuron_index(self):
        """
        Return index of neuron specified in the term
        """
        return self.neuron.get_index()

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
    Represent IF-THEN rule with premise as conjunction of terms and conclusion. Immutable and Hashable.
    """
    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise: Set[Term], conclusion: Union[Term, Conclusion]):
        super(Rule, self).__setattr__('premise', premise)
        super(Rule, self).__setattr__('conclusion', conclusion)

    def __setattr__(self, name, value):
        msg = "'%s' is immutable, can't modify %s" % (self.__class__,
                                                      name)
        raise AttributeError(msg)

    def __str__(self):
        conj_terms = ' AND '.join([str(term) for term in self.premise])
        return "IF " + str(conj_terms) + " THEN " + str(self.conclusion)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.premise == other.premise and
            self.conclusion == other.conclusion
        )

    def __hash__(self): # todo check this
        return hash((self.conclusion))

    def get_premise(self) -> Set[Term]:
        return self.premise

    def get_conclusion(self) -> Union[Term, Conclusion]:
        return self.conclusion

    @classmethod
    def create_initial_rule(self, neuron_layer: int, neuron_index: int, threshold: float, class_name: str):
        return self(premise={Term(Neuron(neuron_layer, neuron_index), '>', threshold)},
                    conclusion=Conclusion(class_name))