from rules.rule import Rule
from rules.ruleset import Ruleset
from rules.C5 import C5

from logic_manipulator.substitute_rules import substitute

def extract_rules(model):
    class_rules = {}

    for output_class in model.class_encodings:
        layer_rulesets = [Ruleset() for _ in range(0, model.n_layers)]

        initial_rule = Rule.initial_rule(output_layer=model.n_layers - 1,
                                        neuron_index=output_class.index,
                                        class_name=output_class.name,
                                        threshold=0.5)

        output_layer = model.n_layers-1
        layer_rulesets[output_layer].add_rules({initial_rule})

        # Extract layer-wise rules
        for hidden_layer in reversed(range(0, output_layer)):
            predictors = model.get_layer_activations(layer_index=hidden_layer)


            term_confidences = layer_rulesets[hidden_layer+1].get_terms_with_conf_from_rule_premises()
            terms = term_confidences.keys()

            for term in terms:
                target = term.apply(model.get_layer_activations_of_neuron(layer_index=hidden_layer + 1,
                                                                          neuron_index=term.get_neuron_index()))

                prior_rule_confidence = term_confidences[term]
                rule_conclusion_map = {True: term, False: term.negate()}
                layer_rulesets[hidden_layer].add_rules(C5(x=predictors, y=target,
                                                          rule_conclusion_map=rule_conclusion_map,
                                                          prior_rule_confidence=prior_rule_confidence))

        # Merge layer-wise rules
        output_rule = initial_rule
        for hidden_layer in reversed(range(0, output_layer)):
            output_rule = substitute(total_rule=output_rule, intermediate_rules=layer_rulesets[hidden_layer])

        class_rules[output_class] = output_rule

    return class_rules