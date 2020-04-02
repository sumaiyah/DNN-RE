"""
Run 1 iteration of rule extraction
"""
from collections import namedtuple

import memory_profiler
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle

from evaluate_rules.evaluate import evaluate
from evaluate_rules.predict import predict
from model.model import Model
from src import TEMP_DIR, DATASET_INFO, RULE_EXTRACTOR

# Data is made up of X (input), y (target)
DataValues = namedtuple('DataValues', 'X y')


def run(X, y, train_index, test_index, model_file_path):
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
    rules = RULE_EXTRACTOR.run(NN_model)
    end_time, end_memory = time.time(), memory_profiler.memory_usage()[0]

    # Use rules for prediction
    NN_model.set_rules(rules)

    # Rule extraction time and memory usage
    time = end_time - start_time
    memory = end_memory - start_memory

    return rules, time, memory
