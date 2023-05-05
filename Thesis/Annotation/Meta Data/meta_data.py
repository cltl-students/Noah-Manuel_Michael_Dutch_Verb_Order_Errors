import pandas as pd
from collections import Counter

df = pd.read_csv('../Data/erroneous_sentences.tsv', sep='\t', keep_default_na=False)

print(Counter(df['Language']))
