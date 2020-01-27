from rules.rule import Rule
from rules.ruleset import Ruleset
from rules.C5 import C5

from logic_manipulator.substitute_rules import substitute


def extract_rules(model):
    class_rules = {}

    for output_class in model.class_encodings:
        layer_rulesets = [Ruleset() for _ in range(0, model.n_layers)]

        # Initial output layer rule
        output_layer = model.n_layers - 1
        initial_rule = Rule.initial_rule(output_layer=output_layer,
                                         neuron_index=output_class.index,
                                         class_name=output_class.name,
                                         threshold=0.5)
        layer_rulesets[output_layer].add_rules({initial_rule})

        # Extract layer-wise rules
        for hidden_layer in reversed(range(0, output_layer)):
            print('Extracting layer %d rules:' % hidden_layer)
            predictors = model.get_layer_activations(layer_index=hidden_layer)

            term_confidences = layer_rulesets[hidden_layer + 1].get_terms_with_conf_from_rule_premises()
            terms = term_confidences.keys()

            # how many terms iterating over
            for _ in terms:
                print('.', end='', flush=True)
            print()

            for term in terms:
                print('.', end='', flush=True)

                #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                target = term.apply(model.get_layer_activations_of_neuron(layer_index=hidden_layer + 1,
                                                                          neuron_index=term.get_neuron_index()))

                prior_rule_confidence = term_confidences[term]
                rule_conclusion_map = {True: term, False: term.negate()}

                layer_rulesets[hidden_layer].add_rules(C5(x=predictors, y=target,
                                                          rule_conclusion_map=rule_conclusion_map,
                                                          prior_rule_confidence=prior_rule_confidence))
            print('done')

        # Merge layer-wise rules
        output_rule = initial_rule
        for hidden_layer in reversed(range(0, output_layer)):
            print('Merging layer %d rules' % hidden_layer, end=' ', flush=True)
            output_rule = substitute(total_rule=output_rule, intermediate_rules=layer_rulesets[hidden_layer])
            print('done')

        class_rules[output_class] = output_rule

        print(' -------------------------------------------------------------------------------------------------- ')

    return class_rules
