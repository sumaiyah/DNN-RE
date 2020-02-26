import numpy as np

def get_train_and_test_indices(fold_index, fold_indices_path):
    """
    Return train and test indices for a given fold

    File of the form
    fold0
    train 0 1 0 2 ...
    test 3 4 6 ...
    fold1
    train 1 5 2 ...
    test 6 8 9 ...
    ...
    """
    with open(fold_indices_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) >= (3 * fold_index) + 2, 'Error: not enough information in fold indices file'

        train_indices = lines[(fold_index * 3) + 1].split(' ')[1:]
        test_indices = lines[(fold_index * 3) + 2].split(' ')[1:]

        # Convert string indices to ints
        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]

    return train_indices, test_indices


def get_labels(fold_index, label_path) -> np.array:
    """
    <...>_labels.txt store space separated labels for each fold on a new line

    Returns a list of integer labels for a file and specific fold
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) > fold_index, ('Error: not enough fold information in %s' % label_path)

        labels = lines[fold_index].split(' ')

        # Convert string labels to integers
        labels = np.array([int(label) for label in labels])

    return labels
