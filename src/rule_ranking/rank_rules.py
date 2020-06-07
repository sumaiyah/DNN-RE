"""
For each class
    for each rule for that class
        cc= number of training examples classified correctly using the rule
        todo: should this be n classified correctly / n training examples of that class
        ic = numberin correctly classified
        k=4
        rl = rule length i.e. the number of terms in the rule

"""
import pickle

from model.generation.helpers.split_data import load_split_indices, load_data
from rules.clause import ConjunctiveClause
from rules.rule import OutputClass, Rule
from rules.term import Neuron
from src import n_fold_rules_fp, N_FOLDS, N_FOLD_CV_SPLIT_INDICIES_FP, DATASET_INFO, DATA_FP

k = 4


def rank_rule(dnf_rule: Rule, X_train, y_train, use_rl: bool):
    """

    Args:
        dnf_rule: dnf rule for a class
        X_train: train data
        y_train: test data
        use_rl: if true perform RF+HC-CMPR else RF+HC

    Returns:

    """
    # Each run of rule extraction return a DNF rule for each output class
    rule_output = dnf_rule.get_conclusion()

    # Each clause in the dnf rule is considered a rule for this output class
    for clause in dnf_rule.get_premise():
        cc = ic = 0
        rl = len(clause.get_terms())

        # Iterate over all items in the training data
        for i in range(0, len(X_train)):
            # Map of Neuron objects to values from input data. This is the form of data a rule expects
            neuron_to_value_map = {Neuron(layer=0, index=j): X_train[i][j]
                                   for j in range(len(X_train[i]))}

            # if rule predicts the correct output class
            if clause.evaluate(data=neuron_to_value_map) and rule_output.encoding == y_train[i]:
                cc += 1
            else:
                ic += 1

        # Compute rule rank_score
        rank_score = ((cc - ic) / (cc + ic)) + cc / (ic + k)
        if use_rl:
            rank_score += cc / rl

        # print('cc: %d, ic: %d, rl: %d  rankscroe: %f' % (cc, ic, rl, rank_score))

        # Save rank score
        clause.set_rank_score(rank_score)


X, y = load_data(DATASET_INFO, DATA_FP)
for fold in range(0, 1):#):
    # Get train and test data folds
    train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
    X_train, y_train = X[train_index], y[train_index]

    # Load extracted rules from disk
    print('Loading extracted rules from disk for fold %d/%d...' % (fold+1, N_FOLDS), end='', flush=True)
    with open(n_fold_rules_fp(fold), 'rb') as rules_file:
        rules = pickle.load(rules_file)
    print('done')

    all_rules = set()
    for rule in rules:
        for clause in rule.get_premise():
            all_rules.add(Rule(premise={clause}, conclusion=rule.get_conclusion()))
    #     # rank_rule(dnf_rule=rule, X_train=X_train, y_train=y_train, use_rl=False)
    print('done')

    print(len(all_rules))