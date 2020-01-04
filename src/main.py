import pandas as pd

from model.model import Model

from rules.extract import extract_rules


nn = Model()
extract_rules(nn)


