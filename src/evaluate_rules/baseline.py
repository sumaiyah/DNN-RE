"""
Baseline that uses C5 to extract rules in a pedagogical manner
"""
from typing import Set

import pandas as pd

from rules.rule import Rule
from rules.C5 import C5

def extract_rules(input_vals: pd.DataFrame, target: pd.DataFrame) -> Set[Rule]:
    rules = C5(x=input_vals, y=target, prior_rule_confidence=1, rule_conclusion_map={True: 1, False: 0})
    print('n_rules: ', len(rules))
    for rule in rules:
        print(rule)
