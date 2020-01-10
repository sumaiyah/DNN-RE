from rules.C5 import C5
from rules.ruleset import Ruleset
from rules.rule import Rule


def extract_rules(model):
    class_rules = {}

    for output_class in model.class_encodings:
        # Initial rule
        total_rule = Rule.initial_rule(output_layer=model.n_layers - 1,
                                        neuron_index=output_class.index,
                                        class_name=output_class.name,
                                        threshold=0.5)

        for hidden_layer in reversed(range(0, model.n_layers - 1)):
            intermediate_rules = Ruleset()

            predictors = model.get_layer_activations(layer_index=hidden_layer)

            terms = total_rule.get_terms_from_rule_premise()
            for term in terms:
                target = term.apply(model.get_layer_activations_of_neuron(layer_index=hidden_layer + 1,
                                                                             neuron_index=term.get_neuron_index()))

                rule_conclusion_map = {True: term, False: term.negate()}
                intermediate_rules.add_rules(C5(x=predictors, y=target, rule_conclusion_map=rule_conclusion_map))

            total_rule = total_rule.merge(intermediate_rules)

        class_rules[output_class] = total_rule

    return class_rules



