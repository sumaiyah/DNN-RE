from rules.C5 import C5
from rules.ruleset import Ruleset
from rules.rule import Rule

from logic_manipulator.substitute_rules import substitute


def extract_rules(model):
    class_rules = {}

    for output_class in model.class_encodings:
        # Initial rule
        total_rule = Rule.initial_rule(output_layer=model.n_layers - 1,
                                       neuron_index=output_class.index,
                                       class_name=output_class.name,
                                       threshold=0.5)

        output_layer = model.n_layers - 1

        for hidden_layer in reversed(range(0, output_layer)):
            print('Extracting layer %d rules:' % hidden_layer)

            intermediate_rules = Ruleset()

            predictors = model.get_layer_activations(layer_index=hidden_layer)

            term_confidences = total_rule.get_terms_with_conf_from_rule_premises()
            terms = term_confidences.keys()

            for _ in terms:
                print('.', end='', flush=True)
            print()

            for term in terms:
                print('.', end='', flush=True)
                target = term.apply(model.get_layer_activations_of_neuron(layer_index=hidden_layer + 1,
                                                                          neuron_index=term.get_neuron_index()))

                prior_rule_confidence = term_confidences[term]
                rule_conclusion_map = {True: term, False: term.negate()}
                intermediate_rules.add_rules(C5(x=predictors, y=target,
                                                rule_conclusion_map=rule_conclusion_map,
                                                prior_rule_confidence=prior_rule_confidence))

            print('done')
            print('Merging layer %d rules' % hidden_layer, end=' ', flush=True)
            total_rule = substitute(total_rule=total_rule, intermediate_rules=intermediate_rules)
            print('done')

        class_rules[output_class] = total_rule

    return class_rules
