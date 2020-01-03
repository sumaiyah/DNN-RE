from model.model import Model
from rules.ruleset import Ruleset
from rules.rule import Rule
from rules.C5 import C5

def extract_rules(model: Model):
    class_indicies = list(model.class_encodings.keys())

    rulesets = [Ruleset(classification=model.class_encodings[i], n_layers=model.n_layers) for i in class_indicies]

    for output_class_index in class_indicies:
        class_ruleset = rulesets[output_class_index]

        class_ruleset.set_initial_rule(output_layer=model.n_layers-1, class_index=output_class_index)

        for hidden_layer in reversed(range(0, model.n_layers-1)):
            next_layer_terms = class_ruleset.get_terms_from_rule_bodies_in_layer(from_layer_index=hidden_layer+1)

            for term in next_layer_terms:
                predictors = model.get_layer_activations(layer_index=hidden_layer)
                target = term.apply(model.get_layer_activations_of_neuron(layer_index=hidden_layer+1,
                                                               neuron_index=term.get_neuron_index()))

                rule_conclusion_map = {True: term, False: term.negate()}
                class_ruleset.add_rules_to_layer(
                    rules=(C5(x=predictors, y=target, rule_conclusion_map=rule_conclusion_map)),
                    from_layer_index=hidden_layer)

        print(class_ruleset)




def extract_intermediate_rulesets():
    pass

def merge_intermediate_rulesets():
    pass
