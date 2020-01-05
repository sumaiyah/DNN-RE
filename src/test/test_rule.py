from unittest import TestCase

from rules.rule import Rule, ConjunctiveClause
from rules.term import Term, Neuron
from rules.ruleset import Ruleset

class TestRule(TestCase):
    def setUp(self) -> None:
        # IF [(h_1,2 > 0.4) AND (h_1,1 > 0.1)] THEN (h_2,1 > 0.5)
        self.base_rule_1_clause_2_terms = Rule(premise={ConjunctiveClause({Term(Neuron(1, 1), '>', 0.1), Term(Neuron(1, 2), '>', 0.4)})},
                                               conclusion=Term(Neuron(2,1), '>', 0.5))
        self.base_rule_from_term_set = Rule.from_term_set(premise={Term(Neuron(1, 1), '>', 0.1), Term(Neuron(1, 2), '>', 0.4)},
                                                          conclusion=Term(Neuron(2,1), '>', 0.5))

        # IF [(h_1,1 > 0.1)] OR [(h_1,2 > 0.4)] THEN (h_2,1 > 0.5)
        self.base_rule_2_clause_1_term = Rule(premise={ConjunctiveClause({Term(Neuron(1, 1), '>', 0.1)}),
                                                       ConjunctiveClause({Term(Neuron(1, 2), '>', 0.4)})},
                                               conclusion=Term(Neuron(2,1), '>', 0.5))

        # IF [(h_0,9 > 0.4) AND (h_0,2 <= 0.1)] THEN (h_1,1 > 0.1)
        # IF [(h_0,2 > 0.1) AND (h_0,1 > 0.2)] THEN (h_1,2 > 0.4)
        self.ruleset_all_1_clause_2_terms = Ruleset.from_set(
            rules={
                Rule(premise={ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2), Term(Neuron(0, 2), '>', 0.1)})},
                     conclusion=Term(Neuron(1, 2), '>', 0.4)),
                Rule(premise={ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4), Term(Neuron(0, 2), '<=', 0.1)})},
                     conclusion=Term(Neuron(1, 1), '>', 0.1))
            }
        )

        # IF [(h_0,1 > 0.2)] OR [(h_0,2 > 0.1)] THEN (h_1,2 > 0.4)
        # IF [(h_0,9 > 0.4)] OR [(h_0,2 <= 0.1)] THEN (h_1,1 > 0.1)
        self.ruleset_all_2_clause_1_term = Ruleset.from_set(
            rules={
                Rule(premise={ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2)}),
                              ConjunctiveClause({Term(Neuron(0, 2), '>', 0.1)})},
                     conclusion=Term(Neuron(1, 2), '>', 0.4)),
                Rule(premise={ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4)}),
                              ConjunctiveClause({Term(Neuron(0, 2), '<=', 0.1)})},
                     conclusion=Term(Neuron(1, 1), '>', 0.1))
            }
        )

        # IF [(h_0,9 > 0.4)] OR [(h_0,1 > 0.2) AND (h_0,2 > 0.1)] THEN (h_1,2 > 0.4)
        # IF [(h_0,2 <= 0.1)] THEN (h_1,1 > 0.1)
        self.ruleset_mixed_clause_size = Ruleset.from_set(
            rules={
                Rule(premise={ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2), Term(Neuron(0, 2), '>', 0.1)}),
                              ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4)})},
                     conclusion=Term(Neuron(1, 2), '>', 0.4)),
                Rule(premise={ConjunctiveClause({Term(Neuron(0, 2), '<=', 0.1)})},
                     conclusion=Term(Neuron(1, 1), '>', 0.1))
            }
        )


        # todo what should it do if not enough rules given to substitute cases?

    def test_get_terms_from_rule_premise(self):
        terms = {Term(Neuron(0, 1), '>', 0.2),
                Term(Neuron(0, 2), '>', 0.1),
                Term(Neuron(0, 9), '>', 0.4),
                Term(Neuron(0, 2), '<=', 0.1)}
        self.assertEqual(self.ruleset_all_1_clause_2_terms.get_terms_from_rule_premises(), terms)
        self.assertEqual(self.ruleset_all_2_clause_1_term.get_terms_from_rule_premises(), terms)

    def test_from_term_set(self):
        # test if Rule initialsed from the class method from_term_set, initialised correctly
        self.assertEqual(self.base_rule_1_clause_2_terms, self.base_rule_from_term_set)

    # def test_initial_rule(self):
    #     self.fail()

    def test_merge(self):
        # -- Merge rule (1 clause with 2 terms) ----------------------------------------------------------

        # IF [(h_0,2 <= 0.1) AND (h_0,2 > 0.1) AND (h_0,1 > 0.2) AND (h_0,9 > 0.4)] THEN (h_2,1 > 0.5)
        expected_rule = Rule(
            premise={ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2),
                                        Term(Neuron(0, 2), '>', 0.1),
                                        Term(Neuron(0, 9), '>', 0.4),
                                        Term(Neuron(0, 2), '<=', 0.1)})},
            conclusion=Term(Neuron(2,1), '>', 0.5)
        )
        merged_rule = self.base_rule_1_clause_2_terms.merge(self.ruleset_all_1_clause_2_terms)
        self.assertEqual(expected_rule, merged_rule)
        # -------------------------------------------------------------------------
        # IF [(h_0,1 > 0.2) AND (h_0,9 > 0.4)] OR [(h_0,2 <= 0.1) AND (h_0,2 > 0.1)] OR [(h_0,2 <= 0.1) AND (h_0,1 > 0.2)] OR [(h_0,2 > 0.1) AND (h_0,9 > 0.4)] THEN (h_2,1 > 0.5)
        expected_rule = Rule(
            premise={
                ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2), Term(Neuron(0, 9), '>', 0.4)}),
                ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2), Term(Neuron(0, 2), '<=', 0.1)}),
                ConjunctiveClause({Term(Neuron(0, 2), '>', 0.1), Term(Neuron(0, 9), '>', 0.4)}),
                ConjunctiveClause({Term(Neuron(0, 2), '>', 0.1), Term(Neuron(0, 2), '<=', 0.1)})
            },
            conclusion=Term(Neuron(2, 1), '>', 0.5)
        )
        merged_rule = self.base_rule_1_clause_2_terms.merge(self.ruleset_all_2_clause_1_term)
        self.assertEqual(expected_rule, merged_rule)
        # --------------------------------------------------------------------------------------------
        # IF [(h_0,1 > 0.2) AND (h_0,2 > 0.1) AND (h_0,2 <= 0.1)] OR [(h_0,9 > 0.4) AND (h_0,2 <= 0.1)] THEN (h_2,1 > 0.5)
        expected_rule = Rule(
            premise={
            ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4), Term(Neuron(0, 2), '<=', 0.1)}),
            ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2), Term(Neuron(0, 2), '>', 0.1), Term(Neuron(0, 2), '<=', 0.1)})
            },
            conclusion=Term(Neuron(2, 1), '>', 0.5)
        )
        merged_rule = self.base_rule_1_clause_2_terms.merge(self.ruleset_mixed_clause_size)
        self.assertEqual(expected_rule, merged_rule)
        # ------------------------------------------------------------------------------------------------

        # -- Merge rule (2 clauses with 1 term) ----------------------------------------------------------
        # IF [(h_0,1 > 0.2) AND (h_0,2 > 0.1)] OR [(h_0,9 > 0.4) AND (h_0,2 <= 0.1)] THEN (h_2,1 > 0.5)
        expected_rule = Rule(
            premise={ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4),
                                        Term(Neuron(0, 2), '<=', 0.1)}),
                     ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2),
                                        Term(Neuron(0, 2), '>', 0.1)})},
            conclusion=Term(Neuron(2, 1), '>', 0.5)
        )
        merged_rule = self.base_rule_2_clause_1_term.merge(self.ruleset_all_1_clause_2_terms)
        self.assertEqual(expected_rule, merged_rule)
        # -------------------------------------------------------------------------
        # IF [(h_0,1 > 0.2)] OR [(h_0,2 <= 0.1)] OR [(h_0,2 > 0.1)] OR [(h_0,9 > 0.4)] THEN (h_2,1 > 0.5)
        expected_rule = Rule(
            premise={ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2)}),
                     ConjunctiveClause({Term(Neuron(0, 2), '>', 0.1)}),
                     ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4)}),
                     ConjunctiveClause({Term(Neuron(0, 2), '<=', 0.1)})
                     },
            conclusion=Term(Neuron(2, 1), '>', 0.5)
        )
        merged_rule = self.base_rule_2_clause_1_term.merge(self.ruleset_all_2_clause_1_term)
        self.assertEqual(expected_rule, merged_rule)
        # --------------------------------------------------------------------------------------------
        # IF [(h_0,9 > 0.4)] OR [(h_0,2 <= 0.1)] OR [(h_0,1 > 0.2) AND (h_0,2 > 0.1)] THEN (h_2,1 > 0.5)
        expected_rule = Rule(
            premise={ConjunctiveClause({Term(Neuron(0, 1), '>', 0.2), Term(Neuron(0, 2), '>', 0.1)}),
                     ConjunctiveClause({Term(Neuron(0, 9), '>', 0.4)}),
                     ConjunctiveClause({Term(Neuron(0, 2), '<=', 0.1)})},
            conclusion=Term(Neuron(2, 1), '>', 0.5)
        )
        merged_rule = self.base_rule_2_clause_1_term.merge(self.ruleset_mixed_clause_size)
        self.assertEqual(expected_rule, merged_rule)

        # todo a rule with i term or 2 terms anded together?

