from model.model import Model

from extract_rules.deep_red import extract_rules as extract_rules_1
from extract_rules.modified_deep_red import extract_rules as extract_rules_2
from extract_rules.pedagogical import extract_rules as pedagogical_extract_rules

from evaluate_rules.evaluate import evaluate


from collections import namedtuple
ClassEncoding = namedtuple('ClassEncoding', 'name index')  # model output classes


base_path = '../../data/'  # Artificial Dataset 1

data_path = base_path + 'Artif-1/'
# data_path = base_path + 'Artif-2/'

# base_path = 'data/MB-ER/'    # for gene data use train/test.csv
# base_path = 'data/XOR-manual/'     # for manually constructed XOR network
# base_path = 'data/BreastCancer/' # UCI breast cancer dataset
# base_path = 'data/MNIST/'     # MNIST Handwritten dataset
# base_path = 'data/LetterRecognition/'  # Letter Recognition Dataset
# base_path = 'data/  # Artificial Dataset 2

params = dict(
    class_encodings=(ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
    data_path=data_path,
    recompute_layer_activations=True
)

import time
# start_time = time.time()
#
nn = Model(**params)
print('Decompositional Rule Extraction ...................................................................')
nn.set_rules(extract_rules_1(nn))
# t = (time.time() - start_time)
# print("--- Rule Extraction took %s seconds ---" % t)
#
# start_time = time.time()
# acc = evaluate_rules.accuracy(nn)
evaluate(model=nn)
# t = (time.time() - start_time)
# print("--- Calculating Accuracy took %s seconds ---" % t)

# nn = Model(**params)
# start_time = time.time()
# print('Pedagogical Rule Extraction ...................................................................')
# nn.set_rules(pedagogical_extract_rules(nn))
# t = (time.time() - start_time)
# print("--- Rule Extraction took %s seconds ---" % t)
# nn.print_rules()
# start_time = time.time()
# acc = evaluate_rules.accuracy(nn)
# t = (time.time() - start_time)
# print("--- Calculating Accuracy took %s seconds ---" % t)
