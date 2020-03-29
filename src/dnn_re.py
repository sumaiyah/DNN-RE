"""
Run 1 iteration of rule extraction
"""
import pandas as pd
import numpy as np

from evaluate_rules.evaluate import evaluate
from evaluate_rules.predict import predict
from model.model import Model
from keras.models import load_model
from src import TEMP_DIR, DATASET_INFO, RULE_EX_MODE

import memory_profiler
from collections import namedtuple

# Data is made up of X (input), y (target)
DataValues = namedtuple('DataValues', 'X y')

def run(X, y, train_index, test_index, model_file_path, label_file_path):
    import time
    # Split data
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    train_data = DataValues(X=X_train, y=y_train)
    test_data = DataValues(X=X_test, y=y_test)

    # Initialise NN Model object
    NN_model = Model(model_path=model_file_path,
                     output_classes=DATASET_INFO.output_classes,
                     train_data=train_data,
                     test_data=test_data,
                     activations_path=TEMP_DIR + 'activations/')

    # Rule Extraction
    start_time, start_memory = time.time(), memory_profiler.memory_usage()[0]
    rules = RULE_EX_MODE.run(NN_model)
    end_time, end_memory = time.time(), memory_profiler.memory_usage()[0]

    # Use rules for prediction
    NN_model.set_rules(rules)

    # Save labels to labels.csv:
    # label - True data labels
    label_data = {'id': test_index,
                  'true_labels': y_test}
    # label - Neural network data labels. Use NN to predict X_test
    nn_model = load_model(model_file_path)
    nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
    label_data['nn_labels'] = nn_predictions
    # label - Rule extraction labels
    rule_predictions = predict(rules, test_data.X)
    label_data['rule_%s_labels' % RULE_EX_MODE.mode] = rule_predictions
    pd.DataFrame(data=label_data).to_csv(label_file_path, index=False)

    time = end_time - start_time
    memory = end_memory - start_memory

    # Evaluation
    results = evaluate(rules, label_file_path)
    results['time'] = time
    results['memory'] = memory

    return results
