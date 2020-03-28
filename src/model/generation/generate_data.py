"""
1. Split data
"""
import pandas as pd

from model.generation.helpers import split_data
from model.generation import DATASET_DIRECTORY, DATA_PATH, DATASET_INFO


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


def run(split_data_flag=False):
    n_folds = 10
    X, y = load_data(DATASET_INFO, DATA_PATH)

    # Only do this once
    if split_data_flag:
        print('Splitting data. WARNING: only do this once!')
        split_data.train_test_split(X=X, y=y, test_size=0.2)
        split_data.stratified_k_fold(X=X, y=y, n_folds=n_folds)

run()
