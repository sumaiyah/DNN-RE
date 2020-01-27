from model.model import Model

from extract_rules.deep_red import extract_rules as extract_rules_1
from extract_rules.modified_deep_red import extract_rules as extract_rules_2

import evaluate_rules.evaluate as evaluate_rules

from collections import namedtuple

ClassEncoding = namedtuple('ClassEncoding', 'name index')  # model output classes

# base_path = 'data/MB/'    # for gene data use train/test.csv
# base_path = 'data/XOR-manual/'     # for manually constructed XOR network
# base_path = 'data/BreastCancer/' # UCI breast cancer dataset
# base_path = 'data/MNIST/'     # MNIST Handwritten dataset
# base_path = 'data/LetterRecognition/'  # Letter Recognition Dataset
# base_path = 'data/Artif-1/'  # Artificial Dataset 1
base_path = 'data/Artif-2/'  # Artificial Dataset 2

params = dict(
    class_encodings=(ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
    activations_path=base_path,
    train_data_path=base_path + 'train_data.csv',
    test_data_path=base_path + 'test_data.csv',
    model_path=base_path + 'model.h5',
    recompute_layer_activations=True
)

import time

start_time = time.time()

nn = Model(**params)
print('Rule Extraction ...................................................................')
nn.set_rules(extract_rules_1(nn))
t = (time.time() - start_time)
print("--- Rule Extraction took %s seconds ---" % t)

nn.save_rules()
nn.print_rules()

start_time = time.time()
acc = evaluate_rules.accuracy(nn)
t = (time.time() - start_time)
print("--- Calculating Accuracy took %s seconds ---" % t)
