from model.model import Model
from extract_rules.deep_red import extract_rules as extract_rules_1
from extract_rules.modified_deep_red import extract_rules as extract_rules_2
from extract_rules.pedagogical import extract_rules as pedagogical_extract_rules
from evaluate_rules.evaluate import evaluate

import time
import memory_profiler
from collections import namedtuple

ClassEncoding = namedtuple('ClassEncoding', 'name index')  # model output classes

base_path = '../../data/'

# data_path = base_path + 'Artif-1/' # Artificial Dataset 1
# data_path = base_path + 'Artif-2/' # Artificial Dataset 2
data_path = base_path + 'MB-ER/'    # for gene data estrogen receptor
# data_path = base_path + 'BreastCancer/' # UCI breast cancer dataset
# data_path = base_path + 'LetterRecognition/'  # Letter Recognition Dataset
# data_path = base_path + 'MNIST/'     # MNIST Handwritten dataset

params = dict(
    class_encodings=(ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
    data_path=data_path,
    recompute_layer_activations=True
)
nn = Model(**params)

print('Decompositional Rule Extraction ...................................................................')
start_time = time.time()
start_memory = memory_profiler.memory_usage()
nn.set_rules(extract_rules_1(nn))
t = (time.time() - start_time)
m = memory_profiler.memory_usage()[0] - start_memory[0]
print("--- Rule Extraction took %s seconds and used %s Mb to execute ---" % (t, m))
nn.print_rules()
decomp_features = evaluate(model=nn)

start_time = time.time()
start_memory = memory_profiler.memory_usage()
print('Pedagogical Rule Extraction ...................................................................')
nn.set_rules(pedagogical_extract_rules(nn))
t = (time.time() - start_time)
m = memory_profiler.memory_usage()[0] - start_memory[0]
print("--- Rule Extraction took %s seconds and used %s Mb to execute ---" % (t, m))
nn.print_rules()
ped_features = evaluate(model=nn)

# TODO refactor and put this in evaluate()
n_overlap = len(decomp_features.intersection(ped_features))
print('Overlapping features: ', n_overlap)
