import pickle

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical

import dnn_re
from evaluate_rules.evaluate import evaluate
from evaluate_rules.predict import predict
from model.generation import generate_data
from model.generation.helpers.init_dataset_dir import clean_up
from model.generation.helpers.split_data import load_split_indices, load_data
from src import N_FOLDS, DATASET_INFO, DATA_FP, RULE_EXTRACTOR, N_FOLD_RESULTS_FP, LABEL_FP, \
    n_fold_model_fp, n_fold_rules_fp, N_FOLD_CV_SPLIT_INDICIES_FP


def cross_validate_re(X, y, extract_rules_flag=False, evaluate_rules_flag=False):
    # Path to cross validated results

    # Extract rules from model from each fold
    if extract_rules_flag:
        for fold in range(0, N_FOLDS):
            # Path to extracted rules from that fold
            extracted_rules_file_path = n_fold_rules_fp(fold)

            # Get train and test data folds
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)

            # Path to neural network model for this fold
            model_file_path = n_fold_model_fp(fold)
            _, nn_accuracy = load_model(model_file_path).evaluate(X[test_index], to_categorical(y[test_index]))

            # Extract rules
            rules, re_time, re_memory = dnn_re.run(X, y, train_index, test_index, model_file_path)

            # Save rules extracted
            print('Saving fold %d/%d rules extracted...' % (fold, N_FOLDS), end='', flush=True)
            with open(extracted_rules_file_path, 'wb') as rules_file:
                pickle.dump(rules, rules_file)
            print('done')

            # Save rule extraction time and memory usage
            print('Saving fold %d/%d results...' % (fold, N_FOLDS), end='', flush=True)
            # Initialise empty results file
            if fold==0:
                pd.DataFrame(data=[], columns=['fold']).to_csv(N_FOLD_RESULTS_FP, index=False)

            results_df = pd.read_csv(N_FOLD_RESULTS_FP)
            row_index = fold
            results_df.loc[row_index, 'fold'] = fold
            results_df.loc[row_index, 'nn_accuracy'] = nn_accuracy
            results_df.loc[row_index, 're_time'] = re_time
            results_df.loc[row_index, 're_memory'] = re_memory
            results_df.to_csv(N_FOLD_RESULTS_FP, index=False)
            print('done')

    # Compute cross-validated results
    if evaluate_rules_flag:
        for fold in range(0, N_FOLDS):
            # Get train and test data folds
            train_index, test_index = load_split_indices(N_FOLD_CV_SPLIT_INDICIES_FP, fold_index=fold)
            # X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            # Path to neural network model for this fold
            model_file_path = n_fold_model_fp(fold)

            # Load extracted rules from disk
            print('Loading extracted rules from disk for fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            with open(n_fold_rules_fp(fold), 'rb') as rules_file:
                rules = pickle.load(rules_file)
            print('done')

            # Save labels to labels.csv:
            # label - True data labels
            label_data = {'id': test_index,
                          'true_labels': y_test}
            # label - Neural network data labels. Use NN to predict X_test
            nn_model = load_model(model_file_path)
            nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
            label_data['nn_labels'] = nn_predictions
            # label - Rule extraction labels
            rule_predictions = predict(rules, X_test)
            label_data['rule_%s_labels' % RULE_EXTRACTOR.mode] = rule_predictions
            pd.DataFrame(data=label_data).to_csv(LABEL_FP, index=False)

            # Evaluate rules
            print('Evaulating rules extracted from fold %d/%d...' % (fold, N_FOLDS), end='', flush=True)
            re_results = evaluate(rules, LABEL_FP)
            print('done')

            # Save rule extraction evaulation results
            row_index = fold
            results_df = pd.read_csv(N_FOLD_RESULTS_FP)
            results_df.loc[row_index, 're_acc'] = re_results['acc']
            results_df.loc[row_index, 're_fid'] = re_results['fid']
            results_df.loc[row_index, 'output_classes'] = str(re_results['output_classes'])
            results_df.loc[row_index, 're_n_rules_per_class'] = str(re_results['n_rules_per_class'])
            results_df.loc[row_index, 'n_overlapping_features'] = str(re_results['n_overlapping_features'])
            results_df.loc[row_index, 'min_n_terms'] = str(re_results['min_n_terms'])
            results_df.loc[row_index, 'max_n_terms'] = str(re_results['max_n_terms'])
            results_df.loc[row_index, 'av_n_terms_per_rule'] = str(re_results['av_n_terms_per_rule'])
            results_df.to_csv(N_FOLD_RESULTS_FP, index=False)


X, y = load_data(DATASET_INFO, DATA_FP)

# Generate neural networks from which rules are to be extracted
generate_data.run(X=X, y=y,
                  split_data_flag=True,
                  grid_search_flag=False,
                  find_best_initialisation_flag=True,
                  generate_fold_data_flag=True)

# Perform n fold cross validated rule extraction on the dataset
cross_validate_re(X, y, extract_rules_flag=True, evaluate_rules_flag=True)

# Remove files from temp/
clean_up()
