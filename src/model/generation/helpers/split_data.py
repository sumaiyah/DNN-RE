"""
Split data and save indices of the split for reproducibility
"""
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit  # Returns indices unlike train_test_split

from model.generation.helpers.init_dataset_dir import create_directory
from src import CROSS_VAL_DIR, INITIALISATIONS_DIR


def save_split_indices(train_index, test_index, file_path):
    """
    Args:
        train_index: List of train indicies of data
        test_index: List of test indices of data
        file_path: File to save split indices

    Write list of indices to split_indices.txt

    File of the form with a train and test line for each fold
    train 0 1 0 2 ...
    test 3 4 6 ...
    train 1 5 2 ...
    test 6 8 9 ...
    ...

    """
    with open(file_path, 'a') as file:
        file.write('train ' + ' '.join([str(index) for index in train_index]) + '\n')
        file.write('test ' + ' '.join([str(index) for index in test_index]) + '\n')


def load_split_indices(file_path, fold_index=0):
    """
    Args:
        file_path: path to split indices file
        fold_index: index of the fold whose train and test indices you want

    Returns:
        train_index: list of integer indices for train data
        test_index: list of integer indices for test data

    File of the form
    train 0 1 0 2 ...
    test 3 4 6 ...
    train 1 5 2 ...
    test 6 8 9 ...
    ...
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) >= (2 * fold_index) + 2, 'Error: not enough information in fold indices file'

        train_index = lines[(fold_index * 2)].split(' ')[1:]
        test_index = lines[(fold_index * 2)+1].split(' ')[1:]

        # Convert string indices to ints
        train_index = [int(i) for i in train_index]
        test_index = [int(i) for i in test_index]

    return train_index, test_index


def clear_file(file_path):
    """

    Args:
        file_path:

    Clear contents of a file given file path

    """
    if os.path.exists(file_path):
        open(file_path, 'w').close()
        print('Cleared contents of file %s' % file_path)


def stratified_k_fold(X, y, n_folds):
    """

    Args:
        X: input features
        y: target
        n_folds: how many folds to split data into

    Split data into folds and saves indices in cross_valiation/<n>_folds/data_split_indices.txt
    """
    # Make directory or <n>_folds/trained_models
    fold_dir = CROSS_VAL_DIR + '%d_folds/' % n_folds
    create_directory(dir_path=fold_dir)
    create_directory(dir_path=fold_dir + 'trained_models')

    # Initialise split indices file
    split_indices_file_path = fold_dir + 'data_split_indices.txt'
    clear_file(split_indices_file_path)

    # Split data
    skf = StratifiedKFold(n_splits=n_folds, random_state=100)

    # Save indices
    for train_index, test_index in skf.split(X, y):
        save_split_indices(train_index=train_index,
                           test_index=test_index,
                           file_path=split_indices_file_path)

    print('Split data into %d folds.' % n_folds)
    # TODO save true labels??


def train_test_split(X, y, test_size=0.2):
    """

    Args:
        X: input features
        y: target
        test_size: percentage of the data used for testing

    Returns:

    Single train test split of the data used for initilising the neural network
    """

    # Initialise split indices file
    split_indices_file_path = INITIALISATIONS_DIR + 'data_split_indices.txt'
    clear_file(split_indices_file_path)

    # Split data
    rs = ShuffleSplit(n_splits=2, test_size=test_size, random_state=100)

    for train_index, test_index in rs.split(X):
        save_split_indices(train_index=train_index,
                           test_index=test_index,
                           file_path=split_indices_file_path)

        # Only want 1 split
        break

    print('Split data into train/test split for initialisation.')


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