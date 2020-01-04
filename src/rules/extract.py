from model.model import Model
from rules.ruleset import ClassRules
from rules.rule import Rule
from rules.C5 import C5

def extract_rules(model: Model):
    class_indicies = list(model.class_encodings.keys())

    class_rulesets = [ClassRules(classification=model.class_encodings[i], n_layers=model.n_layers) for i in class_indicies]

    for output_class_index in class_indicies:

        class_ruleset = class_rulesets[output_class_index]

        class_ruleset.set_initial_rule(output_layer=model.n_layers-1, class_index=output_class_index)

        # Extract intermediate rules -------------------------------------------------------------------------------
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
        # -----------------------------------------------------------------------------------------------------------
        print(class_ruleset)
        # class_ruleset.merge_layerwise_rules()


# NEW ALG
"""
For each class:
    push initial rule onto stack
    
    for each hidden layer:
        output_rules <- rules.pop()
        terms <- extract terms from output rules
        
        predictors = hidden_layer_activations
        for each term:
            target <- term.apply(neuron_hidden_layer_activations)
            
            union C5(predictors, target)
            
        delete_redundant_terms(current_ruleset)    
        output_ruleset <- output_ruleset.merge(current_ruleset)
        
        delete_unsat_terms(output_ruleset)
        delete_redundant_terms(output_ruleset)
        
    output rule is a single DNF rule made up of a set of conjunctive clauses
    simple rule is rule made up of 1 conjunctive clause and conclusion
"""