# Noah-Manuel Michael
# Created: 31.05.2023
# Last updated: 01.06.2023
# Inspect to what extent simplifying the tree structures renders fewer unique trees

import pandas as pd
from collections import Counter

df = pd.read_csv('Data/dev_correct_simplified.tsv', sep='\t', header=0, encoding='utf-8')


print('Num trees in original parses:')
print(len(Counter(df['tree'])))

print('Num trees in simplified parses:')
print(len(Counter(df['simple_tree'])))

# Next step: write something to get tuples of (partofspeech, position) as input for classifier
