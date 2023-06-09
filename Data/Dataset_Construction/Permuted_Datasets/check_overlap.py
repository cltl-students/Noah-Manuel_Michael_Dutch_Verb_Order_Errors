import pandas as pd

df = pd.read_csv('test_shuffled_random_all_and_verbs.tsv', sep='\t', header=0, encoding='utf-8')

print(len(set(df['spaced']).intersection(set(df['verbs_random_no_punc']))))
