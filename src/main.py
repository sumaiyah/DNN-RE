from model.model import Model

from extract_rules.deep_red import extract_rules as extract_rules_1
from extract_rules.modified_deep_red import extract_rules as extract_rules_2

import evaluate_rules.evaluate as evaluate_rules

from collections import namedtuple
ClassEncoding = namedtuple('ClassEncoding', 'name index')   # model output classes


params = dict(
class_encodings = (ClassEncoding(name='Zero', index=0), ClassEncoding(name='One', index=1)),
activations_path = 'misc_data/',
train_data_path = 'misc_data/XOR_train_data.csv',
test_data_path = 'misc_data/XOR_test_data.csv',
model_path = 'misc_data/XOR.h5',
recompute_layer_activations = False
)

nn = Model(**params)

print('DeepRED ...................................................................')
nn.set_class_rules(extract_rules_1(nn))
nn.print_rules()
evaluate_rules.accuracy(nn)
print('END DeepRED ...............................................................')
print()

print('modified alg ...................................................................')
nn.set_class_rules(extract_rules_2(nn))
nn.print_rules()
evaluate_rules.accuracy(nn)
print('END modified alg ...............................................................')
print()



