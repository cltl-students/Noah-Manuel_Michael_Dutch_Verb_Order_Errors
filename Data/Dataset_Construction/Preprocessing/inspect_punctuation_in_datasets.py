# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Inspect all preprocessed datasets for uniformity of punctuation spacing

import re

with open('../Unpermuted_Datasets/readability_data.tsv', encoding='utf-8') as infile:
    content = infile.read()
    punc_toks_readability = {c for c in re.findall(r'\W', content)}
print(punc_toks_readability)

with open('../Unpermuted_Datasets/lassy_data.tsv', encoding='utf-8') as infile:
    content = infile.read()
    punc_toks_lassy = {c for c in re.findall(r'\W', content)}
print(punc_toks_lassy)

with open('../Unpermuted_Datasets/wainot_data.tsv', encoding='utf-8') as infile:
    content = infile.read()
    punc_toks_wainot = {c for c in re.findall(r'\W', content)}
print(punc_toks_wainot)

with open('../Unpermuted_Datasets/leipzig_data.tsv', encoding='utf-8') as infile:
    content = infile.read()
    punc_toks_leipzig = {c for c in re.findall(r'\W', content)}
print(punc_toks_leipzig)
