"""
Represent a ruleset made up of rules
"""

from typing import Set, Dict, List

from rules.rule import Rule, Term, Union, Conclusion, Neuron

class Ruleset:
    """
    Ruleset stores a set of Rules in DNF
    """

    def __init__(self, rules: Dict = None):
        if rules is None:
            rules = {}
        self.rules = rules

    @classmethod
    def from_set(cls, rules: Set[Rule] = None):
        """
        Convert a set of Rules into a dict mapping a conclusion to a list of Set[Terms] i.e disjunctive premises
        """
        if rules is None:
            rules = set()

        rules_conc_to_premises = {} # Dict[Union[Term, Conclusion], List[Set[Term]]]

        for rule in rules:
            if rule.get_conclusion() in rules_conc_to_premises:
                # Duplicate rules are ignored
                if not rule.get_premise() in rules_conc_to_premises[rule.get_conclusion()]:
                    rules_conc_to_premises[rule.get_conclusion()].append(rule.get_premise())
            else:
                rules_conc_to_premises[rule.get_conclusion()] = [rule.get_premise()]

        return cls(rules=rules_conc_to_premises)

    def add_rules(self, rules: Set[Rule]):
        for rule in rules:
            if rule.get_conclusion() in self.rules:
                # Duplicate rules are ignored
                if not rule.get_premise() in self.rules[rule.get_conclusion()]:
                    self.rules[rule.get_conclusion()].append(rule.get_premise())
            else:
                self.rules[rule.get_conclusion()] = [rule.get_premise()]

    def get_terms_from_rule_bodies(self) -> Set[Term]:
        terms = set()

        for rule_conc in self.rules.keys():
            for premise in self.rules[rule_conc]:
                terms = terms.union(premise)

        return terms

    def get_conclusions(self):
        return set(self.rules.keys())

    def get_premises_given_conclusion(self, conclusion) -> List[Set[Term]]:
        return self.rules[conclusion]

    def __str__(self):
        # ruleset_str = '\n'
        #
        # for rule_conc in self.rules.keys():
        #     for premise in self.rules[rule_conc]:
        #         ruleset_str += str(Rule(premise, rule_conc)) + '\n'
        #
        # ruleset_str += '\n'
        # return ruleset_str

    def merge(self, current_output_rs):
        new_rules = {}

        # for each DNF rule
        for rule_conc in current_output_rs.get_conclusions():
            old_premises = current_output_rs.get_premises_given_conclusion(rule_conc)

            for old_premise in old_premises:
                for term in old_premise:
                    if term in self.get_conclusions():
                        pass
                # for conc in self.get_conclusions():
                #     print(conc, end=' ')
                # print(old_premise)
            # intermediate_conclusions = current_output_rs.get_premises_given_conclusion(rule_conc)
            #
            # # new premises
            # new_premises = []
            # for conclusion in intermediate_conclusions:
            #     if conclusion in self.get_conclusions():
            #         premises = self.get_premises_given_conclusion(conclusion)
            #
            #         for premise in premises:
            #             if premise not in new_premises:
            #                 new_premises.append(premise)
            #
            # new_rules[rule_conc] = new_premises

        rs = Ruleset(new_rules)
        print('hiiiiiiiiiiiiiiiiiiiii')
        print(rs)
        return rs



class ClassRules:
    """
    Represent rules generated for each class. Holds intermediate layer-wise rules
    """
    def __init__(self, classification: str, n_layers):
        self.classification = classification
        self.layer_rulesets = [Ruleset() for _ in range(0, n_layers)]

    def get_layer_rules(self, from_layer_index: int) -> Ruleset:
        return self.layer_rulesets[from_layer_index]

    def add_rules_to_layer(self, rules: Set[Rule], from_layer_index: int):
        self.get_layer_rules(from_layer_index).add_rules(rules)

    def get_terms_from_rule_bodies_in_layer(self, from_layer_index: int) -> Set[Term]:
        return self.get_layer_rules(from_layer_index).get_terms_from_rule_bodies()

    # TODO CHANGE THIS!!
    def set_initial_rule(self, output_layer, class_index):
        initial_rule = Rule.create_initial_rule(neuron_layer=output_layer, neuron_index=class_index,
                                                threshold=0.5, class_name=self.classification)
        self.add_rules_to_layer({initial_rule}, from_layer_index=output_layer)

    def __str__(self):
        rs_str = ''
        layer = 0
        for layer_rs in self.layer_rulesets:
            if layer == (len(self.layer_rulesets) - 1):
                rs_str += 'LAYER ' + str(layer) + ' -> ' + 'OUTPUT' + ': '
            else:
                rs_str += 'LAYER ' + str(layer) + ' -> ' + str(layer + 1) + ': '
            rs_str += str(layer_rs)

            layer += 1
        return rs_str

    def merge_layerwise_rules(self):
        """
        Merge layer rulesets into 1 ruleset describing behaviour of output in terms of the network inputs
        """
        n_layers = len(self.layer_rulesets)

        output_ruleset = self.get_layer_rules(n_layers-1)

        sectolast_ruleset = self.get_layer_rules(n_layers-2)

        print(output_ruleset)
        print(sectolast_ruleset)

        sectolast_ruleset.merge(output_ruleset)
        # for i in reversed(range(0, n_layers-1)):
        #     output_ruleset = self.get_layer_rules(i).merge(output_ruleset)

        print('-----------------------------------------------------------------------------------')
