from collections import namedtuple
from rules.rule import OutputClass
from extract_rules.modified_deep_red_C5 import extract_rules as MOD_DeepRED_C5

DatasetMetaData = namedtuple('DatasetMetaData', 'name target_col output_classes n_inputs n_outputs')

output_classes = (OutputClass(name='y0', encoding=0),
                  OutputClass(name='y1', encoding=1))
DATASET_INFO = DatasetMetaData(name='Artif-2', target_col='y', output_classes=output_classes, n_inputs=5, n_outputs=2)
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