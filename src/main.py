from model.model import Model

from extract_rules.deep_red import extract_rules as extract_rules_1

import evaluate_rules.evaluate as evaluate_rules

from collections import namedtuple

ClassEncoding = namedtuple('ClassEncoding', 'name index')  # model output classes

# base_path = 'data/MB/'    # for gene data use train/test.csv
base_path = 'data/XOR/'     # for XOR data use all_data.csv

params = dict(
    class_encodings=(ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
    activations_path=base_path,
    # train_data_path = base_path+'train_data.csv',
    # test_data_path = base_path+'test_data.csv',
    train_data_path=base_path + 'all_data.csv',
    test_data_path=base_path + 'all_data.csv',
    model_path=base_path + 'model.h5',
    recompute_layer_activations=True
)

import time
start_time = time.time()

nn = Model(**params)
print('Rule Extraction ...................................................................')
nn.set_rules(extract_rules_1(nn))
nn.print_rules()
acc = evaluate_rules.accuracy(nn)
print('Rule Extraction ...............................................................')

t = (time.time() - start_time)
print("--- %s seconds ---" % t)



# todo change ruleset back to other definition to see if it speeds up rule extraction execution
