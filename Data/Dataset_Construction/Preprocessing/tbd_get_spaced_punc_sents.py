# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Format sentences to have spaced punctuation in line with Lassy sentences

import re
import pandas as pd

df_sents = pd.read_csv('../Unpermuted_Datasets/readability_data.tsv', sep='\t', header=0, encoding='utf-8')

sents_in_lassy_format = []

special_chars = [';', ':', '.', '?', '!', ',', '(', ')', '\'', '"', '‘', '’', '“', '”']

for sent in df_sents['original']:
    for char in special_chars:
        sent = re.sub(r'(\S)' + re.escape(char), r'\1 ' + char, sent)
        sent = re.sub(re.escape(char) + r'(\S)', char + ' ' + r'\1', sent)
    sent = re.sub(r'([0-9]) \. ([0-9])', r'\1.\2', sent)
    sents_in_lassy_format.append(sent)

print(sents_in_lassy_format)
