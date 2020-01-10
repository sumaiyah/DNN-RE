from unittest import TestCase

from rules.term import Term, Neuron
from rules.clause import ConjunctiveClause

class TestConjunctiveClause(TestCase):
    def setUp(self) -> None:
        # Satisfiable term sets
        self.terms_1 = {Term(Neuron(0, 2), '>', 0.3),
                   Term(Neuron(0, 4), '<=', 0.2),
                   Term(Neuron(0, 2), '>', 0.2)}
        self.terms_1_necessary = {Term(Neuron(0, 4), '<=', 0.2),
                             Term(Neuron(0, 2), '>', 0.2)}

        self.terms_2 = {Term(Neuron(0, 4), '<=', 0.3),
                   Term(Neuron(0, 4), '<=', 0.2),
                   Term(Neuron(0, 2), '>', 0.2)}
        self.terms_2_necessary = {Term(Neuron(0, 4), '<=', 0.3),
                             Term(Neuron(0, 2), '>', 0.2)}

        # Unsatisfiable term sets
        self.unsat_1_terms = {
            Term(Neuron(0, 2), '>', 0.3),
            Term(Neuron(0, 2), '<=', 0.2)
        }

    def test_get_terms(self):
        clause = ConjunctiveClause(terms=self.terms_1, remove_redundant_terms=False)
        self.assertEqual(clause.get_terms(), self.terms_1)

        clause = ConjunctiveClause(terms=self.terms_2, remove_redundant_terms=False)
        self.assertEqual(clause.get_terms(), self.terms_2)

    # def test_union(self):
    #     self.fail()

    def test_remove_redundant_terms(self):
        clause = ConjunctiveClause(terms=self.terms_1, remove_redundant_terms=True)
        self.assertEqual(clause.get_terms(), self.terms_1_necessary)

        clause = ConjunctiveClause(terms=self.terms_2, remove_redundant_terms=True)
        self.assertEqual(clause.get_terms(), self.terms_2_necessary)

    def test_is_satisfiable(self):
        clause = ConjunctiveClause(terms=self.unsat_1_terms, remove_redundant_terms=True)
        self.assertEqual(clause.is_satisfiable(), False)

        clause = ConjunctiveClause(terms=self.unsat_1_terms, remove_redundant_terms=False)
        self.assertEqual(clause.is_satisfiable(), False)

