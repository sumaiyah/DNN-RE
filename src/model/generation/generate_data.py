"""
Generates neural networks for each of the n folds using the procedure specified to locate optimal neural network
hyper parameters and neural network initialisation
"""
from collections import OrderedDict

import pandas as pd

from model.generation.helpers import split_data, find_best_nn_initialisation
from model.generation.helpers.build_and_train_model import build_and_train_model
from model.generation.helpers.grid_search import grid_search
from model.generation.helpers.init_dataset_dir import clean_up
from model.generation.helpers.split_data import load_split_indices
from src import N_FOLDS, N_FOLD_CV_SPLIT_INDICIES_FP, n_fold_model_fp, N_FOLD_RESULTS_FP, \
    BATCH_SIZE, EPOCHS, LAYER_1, LAYER_2


def run(X, y, split_data_flag=False, grid_search_flag=False, find_best_initialisation_flag=False,
        generate_fold_data_flag=False):
    print(N_FOLDS)
    """

    Args:
        split_data_flag: Split data. Only do this once!
        grid_search_flag: Grid search to find best neural network hyperparameters.
        find_best_initialisation_flag: Find best neural network initialisation
        generate_fold_data_flag: Generate neural networks for each data fold

    """
    # 1. Split data into train and test. Only do this once
    if split_data_flag:
        print('Splitting data. WARNING: only do this once!')
        split_data.train_test_split(X=X, y=y, test_size=0.2)
        split_data.stratified_k_fold(X=X, y=y, n_folds=N_FOLDS)

    # 2. Grid search over neural network hyper params to find optimal neural network hyperparameters
    if grid_search_flag:
        print('Performing grid search over hyper paramters WARNING this is very expensive')
        grid_search(X=X, y=y)

    # TODO change this to read best grid search hyperparameters from disk
    nn_hyperparameters = OrderedDict(batch_size=BATCH_SIZE,
                                     epochs=EPOCHS,
                                     layer_1=LAYER_1,
                                     layer_2=LAYER_2)

    # 3. Initialise 5 neural networks using 1 train test split
    # Pick initialisation that yields the smallest ruleset
    if find_best_initialisation_flag:
        find_best_nn_initialisation.run(X, y, nn_hyperparameters)

    # 4. Build neural network for each fold using best initialisation found above
    if generate_fold_data_flag:
        for fold in range(0, N_FOLDS):
            print('Training model %d/%d' % (fold, N_FOLDS))

            # Split data using precomputed split indices
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
            X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

            # Model to be stored in <dataset name>\cross_validation\<n>_folds\trained_models\
            model_file_path = n_fold_model_fp(fold)
            build_and_train_model(X_train, y_train, X_test, y_test,
                                  **nn_hyperparameters,
                                  model_file_path=model_file_path,
                                  with_best_initilisation_flag=True)
    # Remove files from temp/
    clean_up()
