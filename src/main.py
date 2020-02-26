from collections import namedtuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model.model import Model
import time
import memory_profiler

from rules.rule import OutputClass
from configurations import get_configuration

from extract_rules.deep_red_C5 import extract_rules as DeepRED_C5
from extract_rules.modified_deep_red_C5 import extract_rules as MOD_DeepRED_C5
from extract_rules.pedagogical import extract_rules as Pedagogical

from evaluate_rules.predict import predict
from evaluate_rules.evaluate import evaluate

from get_fold_data import get_train_and_test_indices, get_labels

# Column names
ColNames = namedtuple('Columns', 'features target')

# Data for a fold contains X (input) and y (target)
FoldData = namedtuple('FoldData', 'X y')

# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', 'mode run')

# -------------------------------------- Set Parameters -------------------------------------------
# Chosen dataset for Rule Extraction
# dataset_name = 'Artif-2'  # Artificial Dataset 1
# target_col_name, output_classes = get_configuration(dataset_name)
dataset_name = 'Artif-2'  # Artificial Dataset 1
target_col_name = 'y'
output_classes = (OutputClass(name='y0', encoding=0),
                  OutputClass(name='y1', encoding=1))

# Algorithm for extracting rules, DeepRED, MOD_DeepRED_C5, Pedagogical
# Ensure change both mode, and run accordingly
# RuleEx = RuleExMode(mode='Pedagogical', run=Pedagogical)
# RuleEx = RuleExMode(mode='DeepRED_C5', run=DeepRED_C5)
RuleEx = RuleExMode(mode='MOD_DeepRED_C5', run=MOD_DeepRED_C5)

# ------------------------------------- Set data locations and initialise files -----------------------------------
# Main data location
BASE_PATH = '../../data/%s/' % dataset_name

MODEL_PATH = BASE_PATH + 'models/'
LABEL_PATH = BASE_PATH + 'labels/'
ACTIVATIONS_PATH = BASE_PATH + 'activations/'

# File paths
INFORMATION_PATH = BASE_PATH + 'information.txt'
FOLD_INDICES_PATH = BASE_PATH + 'fold_indices.txt'

RULE_PREDICTIONS_PATH = LABEL_PATH + '%s_labels.txt' % RuleEx.mode
TRUE_LABELS_PATH = LABEL_PATH + 'TRUE_labels.txt'
NN_LABELS_PATH = LABEL_PATH + 'NN_labels.txt'

# Clear past rule prediction data
open(RULE_PREDICTIONS_PATH, 'w').close()

# Initialise information path with rule ex metadata
with open(INFORMATION_PATH, 'a') as file:
    file.write('\nRule Extraction: \n')
    file.write('Mode: %s \n' % RuleEx.mode)

# ------------------------ ----------- Perform Cross-Validated Rule Extraction ------------------------------------
# Retrieve data for the dataset specified
data = pd.read_csv(BASE_PATH + 'data.csv')

# Retrieve relevant data, split into input features (X) and target column (y)
X = data.drop([target_col_name], axis=1).values
y = data[target_col_name].values

# List of input feature names
feature_col_names = list(data.columns)
feature_col_names.remove(target_col_name)

# Perform Rule Extraction over each fold
n_folds = 5
for fold_index in range(0, n_folds):
    with open(INFORMATION_PATH, 'a') as file:
        file.write('\n Fold %d \n' % fold_index)

    # Get pre-specified train/test indices
    train_indices, test_indices = get_train_and_test_indices(fold_index, fold_indices_path=FOLD_INDICES_PATH)

    # Train and Test data are defined for each fold
    train_data = FoldData(X=X[train_indices], y=y[train_indices])
    test_data = FoldData(X=X[test_indices], y=y[test_indices])

    # Instantiate Model for each fold
    NN_model = Model(model_path=MODEL_PATH + ('fold_%d.h5' % fold_index),
                     col_names=ColNames(features=feature_col_names, target=target_col_name),
                     output_classes=output_classes,
                     train_data=train_data,
                     test_data=test_data,
                     activations_path=ACTIVATIONS_PATH,
                     recompute_layer_activations=True)

    # Extract Rules using chosen algorithm
    start_time, start_memory = time.time(), memory_profiler.memory_usage()[0]
    rules = RuleEx.run(NN_model)
    NN_model.set_rules(rules)
    end_time, end_memory = time.time(), memory_profiler.memory_usage()[0]

    # Use ruleset to classify test data (X_test)
    # Save rule-based predictions to {Decomp/Pedagogical/DeepRED}_labels.txt
    rule_based_predictions = predict(NN_model, data=NN_model.test_data.X)
    with open(RULE_PREDICTIONS_PATH, 'a') as file:
        file.write(' '.join([str(pred) for pred in rule_based_predictions]))
        file.write('\n')

    # Save runtime information to disk i.e. time, RAM usage
    with open(INFORMATION_PATH, 'a') as file:
        file.write('Time: %s seconds ' % (end_time - start_time))
        file.write('Memory: %s Mb \n' % (end_memory - start_memory))

    # Evaluate predictions and save evaluation metrics to disk
    predicted_labels = get_labels(fold_index, label_path=RULE_PREDICTIONS_PATH)
    network_labels = get_labels(fold_index, label_path=NN_LABELS_PATH)
    true_labels = get_labels(fold_index, label_path=TRUE_LABELS_PATH)

    evaluate(model=NN_model,
             predicted_labels=predicted_labels,
             network_labels=network_labels,
             true_labels=true_labels,
             runtime_data_path=INFORMATION_PATH)

    # Print rules
    NN_model.print_rules()

    print('-------- end of fold %d -------------' % fold_index)

with open(INFORMATION_PATH, 'a') as file:
    file.write('------------------------------------------------------------- \n')