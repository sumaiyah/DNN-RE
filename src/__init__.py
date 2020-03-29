from collections import namedtuple

from configurations import get_configuration
from extract_rules.modified_deep_red_C5 import extract_rules as MOD_DeepRED_C5

DATASET_INFO = get_configuration(dataset_name='MB-GE-ER')
N_FOLDS = 10

DATASET_DIRECTORY = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/DNN-RE-data/%s/' % DATASET_INFO.name

CROSS_VAL_DIR = DATASET_DIRECTORY + 'cross_validation/'
INITIALISATIONS_DIR = DATASET_DIRECTORY + 'neural_network_initialisation/'
RESULTS_DIR = DATASET_DIRECTORY + 'results/'

DATA_PATH = DATASET_DIRECTORY + 'data.csv'

TEMP_DIR = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/DNN-RE/src/temp/'

# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', 'mode run')
RULE_EX_MODE = RuleExMode(mode='MOD_DeepRED_C5', run=MOD_DeepRED_C5)