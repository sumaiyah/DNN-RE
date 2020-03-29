import pandas as pd

import dnn_re
from model.generation import generate_data
from model.generation.helpers.init_dataset_dir import clean_up
from model.generation.helpers.split_data import load_split_indices, load_data
from src import N_FOLDS, CROSS_VAL_DIR, DATASET_INFO, DATA_PATH, TEMP_DIR, RESULTS_DIR, RULE_EX_MODE

def cross_validate_re(X, y):
    # Path to labels saved for each fold
    label_file_path = TEMP_DIR + 'labels.csv'

    # Path to cross validated results
    results_file_path = RESULTS_DIR + 'cross_val_rule_ex_results.csv'

    # Extract rules from model from each fold
    for fold in range(0, N_FOLDS):
        n_fold_cross_val_dir = CROSS_VAL_DIR + '%d_folds/' % N_FOLDS

        # Get fold data and split data using precomputed split indices
        data_split_indices_file_path = n_fold_cross_val_dir + 'data_split_indices.txt'
        train_index, test_index = load_split_indices(data_split_indices_file_path, fold_index=fold)

        # Path to neural network model for this fold
        model_file_path = n_fold_cross_val_dir + 'trained_models/' + 'model_%d.h5' % fold

        # Extract rules
        re_results = dnn_re.run(X, y, train_index, test_index, model_file_path, label_file_path)

        # Save rule extraction results results
        print('Saving fold %d results...' % fold, end='', flush=True)
        results_df = pd.read_csv(results_file_path)
        row_index = len(results_df) + fold
        results_df.loc[row_index, 'run'] = fold
        results_df.loc[row_index, 're_mode'] = RULE_EX_MODE.mode
        results_df.loc[row_index, 're_acc'] = re_results['acc']
        results_df.loc[row_index, 're_fid'] = re_results['fid']
        results_df.loc[row_index, 're_time'] = re_results['time']
        results_df.loc[row_index, 're_memory'] = re_results['memory']
        results_df.loc[row_index, 're_n_rules_per_class'] = str(re_results['n_rules_per_class'])

        results_df.to_csv(results_file_path, index=False)
        print('done')

    # TODO Compute cross-validated results
    results_df = pd.read_csv(results_file_path)


X, y = load_data(DATASET_INFO, DATA_PATH)

# Generate neural networks from which rules are to be extracted
generate_data.run(X=X, y=y,
                  split_data_flag=False,
                  grid_search_flag=False,
                  find_best_initialisation_flag=False,
                  generate_fold_data_flag=False)

# Perform n fold cross validated rule extraction on the dataset
cross_validate_re(X, y)

# Remove files from temp/
clean_up()