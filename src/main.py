from collections import namedtuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model.model import Model
import time
import memory_profiler

from extract_rules.deep_red import extract_rules as DeepRED
from extract_rules.pedagogical import extract_rules as Pedagogical

from evaluate_rules.predict import predict

# Encode each output class with an index
ClassEncoding = namedtuple('ClassEncoding', 'name index')

# Column names
ColNames = namedtuple('Columns', 'features target')

# Data for a fold contains X (input) and y (target)
FoldData = namedtuple('FoldData', 'X y')

# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', 'mode run')

# -------------------------------------- Set Parameters -------------------------------------------
# Main data location
base_data_path = '../../data/'

# Chosen dataset for Rule Extraction
dataset_name = 'Artif-1' # Artificial Dataset 1

# Target column name and subsequent class names
target_col_name = 'y'
class_encodings = (ClassEncoding('y0', index=0),
                   ClassEncoding('y1', index=1))

# Algorithm for extracting rules, DeepRED, Decomp, Pedagogical
# Ensure change both name, and run accordingly
RE = RuleExMode(mode='Pedagogical', run=Pedagogical)
# ---------------------------------------- Pre-process data ----------------------------------------
# Retrieve data for the dataset specified
data_path = base_data_path + dataset_name + '/'
data = pd.read_csv(data_path + 'data.csv')

# Clear past rule extraction labels
open(data_path + ('%s_labels.txt' % RE.mode), 'w').close()
open(data_path + 'runtime.txt', 'w').close()

# List of input feature names
feature_col_names = list(data.columns)
feature_col_names.remove(target_col_name)

# Retrieve relevant data, split into input features (X) and target column (y)
X = data[feature_col_names].values
y = data[target_col_name].values
# --------------------------------------------------------------------------------------------------

fold_index = 0
skf = StratifiedKFold(n_splits=5, random_state=1)
for train_index, test_index in skf.split(X, y):

    # Train and Test data are defined for each fold
    train_data = FoldData(X=X[train_index], y=y[train_index])
    test_data = FoldData(X=X[test_index], y=y[test_index])

    # Instantiate Model for each fold
    NN_model = Model(model_path=data_path + ('model_fold_%d.h5' % fold_index),
                     col_names=ColNames(features=feature_col_names, target=target_col_name),
                     class_encodings=class_encodings,
                     train_data=train_data,
                     test_data=test_data,
                     activations_path=data_path,
                     recompute_layer_activations=True)

    # Extract Rules using chosen algorithm
    start_time, start_memory = time.time(),  memory_profiler.memory_usage()[0]
    NN_model.set_rules(RE.run(NN_model))
    end_time, end_memory = time.time(), memory_profiler.memory_usage()[0]

    # Use ruleset to classify test data (X_test)
    # Save rule-based predictions to {Decomp/Pedagogical/DeepRED}_labels.txt
    rule_based_predictions = predict(NN_model, data=NN_model.test_data.X)
    with open(data_path + ('%s_labels.txt' % RE.mode), 'a') as file:
        file.write(' '.join([str(pred) for pred in rule_based_predictions]))
        file.write('\n')

    # Save runtime information to disk i.e. time, RAM usage
    with open(data_path + 'runtime.txt', 'a') as file:
        file.write('Fold %d \n' % fold_index)
        file.write('Time: %s seconds \n' % (end_time - start_time))
        file.write('Memory: %s Mb \n' % (end_memory - start_memory))

    print('-------- end of fold %d -------------' % fold_index)
    fold_index += 1

# Evaluate predictions

