"""
1. Split data
"""
from collections import OrderedDict

import pandas as pd

from model.generation.helpers import split_data
from model.generation import DATA_PATH, DATASET_INFO
from model.generation.helpers.generate_data import grid_search
from model.generation import find_best_initialisation
from model.generation.helpers.init_dataset_dir import clean_up


def load_data(dataset_info, data_path):
    """

    Args:
        dataset_info: meta data about dataset e.g. name, target col
        data_path: path to data.csv

    Returns:
        X: data input features
        y: data target
    """
    data = pd.read_csv(data_path)

    X = data.drop([dataset_info.target_col], axis=1).values
    y = data[dataset_info.target_col].values

    return X, y


def run(split_data_flag=False, grid_search_flag=False, find_best_initialisation_flag=False):
    n_folds = 10
    X, y = load_data(DATASET_INFO, DATA_PATH)

    # Split data into train and test. Only do this once
    if split_data_flag:
        print('Splitting data. WARNING: only do this once!')
        split_data.train_test_split(X=X, y=y, test_size=0.2)
        split_data.stratified_k_fold(X=X, y=y, n_folds=n_folds)

    # Grid search over neural network hyper params to find optimal
    if grid_search_flag:
        print('Performing grid search over hyper paramters WARNING this is very expensive')
        grid_search(X=X, y=y)

    # Initialise 5 neural networks using 1 train test split
    # Pick initialisation that yields the smallest ruleset
    if find_best_initialisation_flag:
        # TODO change this to read best grid search hyperparameters from disk
        batch_size, epochs, layer_1, layer_2 = 10, 50, 10, 5
        hyperparameters = OrderedDict(batch_size=batch_size, epochs=epochs, layer_1=layer_1, layer_2=layer_2)
        find_best_initialisation.run(X, y, hyperparameters)

    clean_up()


run(find_best_initialisation_flag=True)
