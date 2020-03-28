from collections import namedtuple
DatasetMetaData = namedtuple('DatasetMetaData', 'name target_col')
DATASET_INFO = DatasetMetaData(name='Artif-2', target_col='y')

DATASET_DIRECTORY = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/DNN-RE-data/%s/' % DATASET_INFO.name

CROSS_VAL_DIR = DATASET_DIRECTORY + 'cross_validation/'
INITIALISATIONS_DIR = DATASET_DIRECTORY + 'neural_network_initialisation/'

DATA_PATH = DATASET_DIRECTORY + 'data.csv'
