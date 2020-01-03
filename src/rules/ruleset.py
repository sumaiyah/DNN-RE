"""
Represent a ruleset made up of rules
"""

from typing import Set, Dict, List

from rules.rule import Rule, Term, Union, Conclusion

class LayerRuleset:
    """
    Represent layerwise ruleset
    """
    def __init__(self, rules: Set[Rule] = None):
        if rules is None:
            rules = set()

        self.rules = self.rule_set_to_dict(rules)

    def rule_set_to_dict(self, rules: Set[Rule]):
        """
        Store set of rules as a dictionary mapping conclusion to all possible term sets. Easier to merge rules later
        """
        rules_conc_to_premise = {}

        for rule in rules:
            if rule.get_conclusion() in rules_conc_to_premise:
                rules_conc_to_premise[rule.get_conclusion()].append(rule.get_premise())
            else:
                rules_conc_to_premise[rule.get_conclusion()] = [rule.get_premise()]

        return rules_conc_to_premise

    def add_rule(self, rule: Rule):
        """
        Add rule to ruleset
        """
        if rule.get_conclusion() in self.rules:

            # Duplicate rules are ignored
            if not rule.get_premise() in self.rules[rule.get_conclusion()]:
                self.rules[rule.get_conclusion()].append(rule.get_premise())
        else:
            self.rules[rule.get_conclusion()] = [rule.get_premise()]

    def add_rules(self, rules: Set[Rule]):
        for rule in rules:
            self.add_rule(rule)

    def get_terms_from_rule_bodies(self) -> Set[Term]:
        terms = set()

        for rule_conc in self.rules.keys():
            for term_set in self.rules[rule_conc]:
                terms = terms.union(term_set)

        return terms

    def __str__(self):
        rules: Set[Rule] = {Rule(premise, conclusion)
                            for conclusion, premise_list in self.rules.items()
                            for premise in premise_list}

        rules_str = '\n'
        for rule in rules:
            rules_str += str(rule) + '\n'
        rules_str += '\n'
        return rules_str

class Ruleset:
    """
    Represent class ruleset
    """
    def __init__(self, classification: str, n_layers):
        self.classification = classification
        self.layer_rulesets = [LayerRuleset() for _ in range(0, n_layers)]

    def add_rules_to_layer(self, rules: Set[Rule], from_layer_index: int):
        layer_ruleset = self.layer_rulesets[from_layer_index]
        layer_ruleset.add_rules(rules)

    def get_terms_from_rule_bodies_in_layer(self, from_layer_index: int) -> Set[Term]:
        layer_ruleset = self.layer_rulesets[from_layer_index]
        return layer_ruleset.get_terms_from_rule_bodies()

    def set_initial_rule(self, output_layer, class_index):
        initial_rule = Rule.create_initial_rule(neuron_layer=output_layer, neuron_index=class_index,
                             threshold=0.5, class_name=self.classification)
        self.add_rules_to_layer({initial_rule}, from_layer_index=output_layer)

    def __str__(self):
        rs_str = ''
        layer = 0
        for layer_rs in self.layer_rulesets:
            if layer==(len(self.layer_rulesets)-1):
                rs_str += 'LAYER ' + str(layer) + ' -> ' + 'OUTPUT' + ': '
            else:
                rs_str += 'LAYER ' + str(layer) + ' -> ' + str(layer+1) + ': '
            rs_str += str(layer_rs)

            layer += 1
        return rs_str

