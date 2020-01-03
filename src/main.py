import pandas as pd

from model.model import Model

from rules.extract import extract_rules

train_data = pd.read_csv('misc_data/XOR_train_data.csv')

# target_labels=['Zero', 'One']

nn = Model()
extract_rules(nn)



# re = RuleExtractor(nn)
# re.extract_intermediate_rules()

# test = pd.read_csv('../misc_data/XOR_test_data.csv'),
