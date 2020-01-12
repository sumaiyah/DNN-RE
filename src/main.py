from model.model import Model

from extract_rules.deep_red import extract_rules as extract_rules_1
from extract_rules.modified_deep_red import extract_rules as extract_rules_2

import evaluate_rules.evaluate as evaluate_rules

from collections import namedtuple
ClassEncoding = namedtuple('ClassEncoding', 'name index')   # model output classes


# params = dict(
# class_encodings = (ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
# activations_path = 'misc_data/XOR/',
# train_data_path = 'misc_data/XOR/train_data.csv',
# test_data_path = 'misc_data/XOR/test_data.csv',
# model_path = 'misc_data/XOR/model_old.h5',
# recompute_layer_activations = False
# )
# base_path = 'misc_data/MB/'
base_path = 'misc_data/XOR/'

params = dict(
class_encodings = (ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
activations_path = base_path,
# train_data_path = base_path+'train_data.csv',
train_data_path = base_path+'train_data.csv',
test_data_path = base_path+'test_data.csv',
model_path = base_path+'model.h5',
recompute_layer_activations = True
)


nn = Model(**params)

print('DeepRED ...................................................................')
nn.set_output_class_to_dnf(extract_rules_1(nn))
nn.print_rules()
evaluate_rules.accuracy(nn)
print('END DeepRED ...............................................................')


# print('modified alg ...................................................................')
# nn.set_class_rules(extract_rules_2(nn))
# nn.print_rules()
# evaluate_rules.accuracy(nn)
# print('END modified alg ...............................................................')
# print()



