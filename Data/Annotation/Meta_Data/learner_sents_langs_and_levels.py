# Noah-Manuel Michael
# Created: 26.05.2023
# Last updated: 26.05.2023
# Get the languages levels for the unique learner sents

import pandas as pd
from collections import Counter

df = pd.read_csv('../Data/final_annotated_data.tsv', header=0, encoding='utf-8', sep='\t')

languages = []
levels = []
sent_cache = set()

for i, row in df.iterrows():
    if not row['Normalized'] in sent_cache:
        languages.append(row['Language'])
        levels.append(row['Level'])
        sent_cache.add(row['Normalized'])

print(len(languages))
print(Counter(languages))

print(len(levels))
print(Counter(levels))