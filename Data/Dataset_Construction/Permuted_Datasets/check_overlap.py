# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 09.06.2023
# check overlap in tree structures

import pandas as pd

df = pd.read_csv('test_shuffled_random_all_and_verbs.tsv', sep='\t', header=0, encoding='utf-8')

print(len(set(df['no_punc']).intersection(set(df['verbs_random_no_punc']))))
