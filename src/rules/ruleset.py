"""
Represent a ruleset made up of rules
"""

from typing import Set, Dict

from rules.term import Term
from rules.clause import ConjunctiveClause
from rules.rule import Rule

def add_rules_to_dict(rules: Set[Rule], rule_conc_to_premises: Dict):
    # Adds rules to dictionary that mapping each conclusions to a set of ConjunctiveClauses
    for rule in rules:
        if rule.get_conclusion() in rule_conc_to_premises:
            rule_conc_to_premises[rule.get_conclusion()] = \
                rule_conc_to_premises[rule.get_conclusion()].union(rule.get_premise())
        else:
            rule_conc_to_premises[rule.get_conclusion()] = rule.get_premise()

class Ruleset:
    """
    Represents a set of disjunctive rules
    """
    def __init__(self, rules: Dict = None):
        if rules is None:
            rules = {}

        self.rules = rules

    @classmethod
    def from_set(cls, rules: Set[Rule] = None) -> 'Ruleset':
        """
        Initialise from a set of rules, convert into a dictionary mapping conclusions to sets of ConjunctiveClauses
        """
        if rules is None:
            rules = set()

        rules_dict = {}
        add_rules_to_dict(rules=rules, rule_conc_to_premises=rules_dict)

        return cls(rules=rules_dict)

    def get_rule_premises_by_conclusion(self, conclusion) -> Set[ConjunctiveClause]:
        """
        Return a set of conjunctive clauses that all imply a given conclusion
        """
        return self.rules[conclusion] if conclusion in self.rules else set()

    def add_rules(self, rules: Set[Rule]):
        """
        Add rules to the ruleset dictionary of rules
        """
        add_rules_to_dict(rules, rule_conc_to_premises=self.rules)

    # def get_terms_from_rule_premises(self) -> Set[Term]:
    def get_terms_from_rule_premises(self) -> Dict:
        """
        Return all the terms present in the bodies of all the rules in the ruleset
        """

        # Values in rules dictionary are all the sets of conjunctive clauses
        # terms = set()
        # for conjunctive_clause_set in self.rules.values():
        #     for clause in conjunctive_clause_set:
        #         terms = terms.union(clause.get_terms())
        # return terms

        # Get terms and MAX clause confidence for each term
        term_confidences = {}
        for conjunctive_clause_set in self.rules.values():
            for clause in conjunctive_clause_set:
                for term in clause.get_terms():
                    if term in term_confidences:
                        term_confidences[term] = max(clause.get_confidence(), term_confidences[term])
                    else:
                        term_confidences[term] = clause.get_confidence()
        return term_confidences

    def rule_dnf_str(self):
        # print rules in DNF form
        ruleset_str = '\n'

        for rule_conc in self.rules.keys():
            ruleset_str += str(Rule(premise=self.rules[rule_conc], conclusion=rule_conc)) + '\n'

        ruleset_str += '\n'
        return ruleset_str

    def rules_all_simple_str(self):
        # prints all rules separately. Each rule premise is conjunction of terms

        # Get all rules individually
        rules = set()
        for rule_conc in self.rules.keys():
            for clause in self.rules[rule_conc]:
                rules.add(Rule(premise={clause}, conclusion=rule_conc))

        # print str
        ruleset_str = '\n'
        for rule in rules:
            ruleset_str += str(rule) + '\n'

        ruleset_str += '\n'
        return ruleset_str



