import pandas as pd

from model.model import Model

from extract_rules.modified_deep_red import extract_rules as extract_rules_2
from extract_rules.deep_red import extract_rules as extract_rules_1


nn = Model()
rule1 = extract_rules_1(nn)
rule2 = extract_rules_2(nn)
print((rule1))
print((rule2))

print(rule1==rule2)

