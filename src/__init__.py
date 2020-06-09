from collections import namedtuple

from configurations import get_configuration
from extract_rules.modified_deep_red_C5 import extract_rules as MOD_DeepRED_C5

DATASET_INFO = get_configuration(dataset_name='Artif-1')
N_FOLDS = 10

# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', 'mode run')
RULE_EXTRACTOR = RuleExMode(mode='MOD_DeepRED_C5', run=MOD_DeepRED_C5)

BATCH_SIZE = 10
EPOCHS = 50
LAYER_1 = 10
LAYER_2 = 5

# --------------------------------------------- File paths -----------------------------------------------------------
# NB: FP/fp = file path, DP/dp = directory path

DATASET_DP = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/DNN-RE-data/%s/' % DATASET_INFO.name
DATA_FP = DATASET_DP + 'data.csv'

# <dataset_name>/cross_validation/<n>_folds/
CV_DP = DATASET_DP + 'cross_validation/'
N_FOLD_CV_DP = CV_DP + '%d_folds/' % N_FOLDS
N_FOLD_CV_SPLIT_INDICIES_FP = N_FOLD_CV_DP + 'data_split_indices.txt'

# <dataset_name>/cross_validation/<n>_folds/rule_extraction/<rule_ex_mode>/rules_extracted/
N_FOLD_RULE_EX_MODE_DP = N_FOLD_CV_DP + 'rule_extraction/' + RULE_EXTRACTOR.mode + '/'
N_FOLD_RESULTS_FP = N_FOLD_RULE_EX_MODE_DP + 'results.csv'
N_FOLD_RULES_DP = N_FOLD_RULE_EX_MODE_DP + 'rules_extracted/'
n_fold_rules_fp = lambda fold: N_FOLD_RULES_DP + 'fold_%d.rules' % fold

# <dataset_name>/cross_validation/<n>_folds/trained_models/
N_FOLD_MODELS_DP = N_FOLD_CV_DP + 'trained_models/'
n_fold_model_fp = lambda fold: N_FOLD_MODELS_DP + 'fold_%d_model.h5' % fold

# <dataset_name>/neural_network_initialisation/
NN_INIT_DP = DATASET_DP + 'neural_network_initialisation/'
NN_INIT_GRID_RESULTS_FP = NN_INIT_DP + 'grid_search_results.txt'
NN_INIT_SPLIT_INDICES_FP = NN_INIT_DP + 'data_split_indices.txt'
NN_INIT_RE_RESULTS_FP = NN_INIT_DP + 're_results.csv'
BEST_NN_INIT_FP = NN_INIT_DP + 'best_initialisation.h5'

# Store temporary files during program execution
TEMP_DIR = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/DNN-RE/src/temp/'
LABEL_FP = TEMP_DIR + 'labels.csv'
