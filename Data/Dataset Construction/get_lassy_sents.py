# Noah-Manuel Michael
# Created: 21.05.2023
# Last updated: 21.05.2023
# Get Lasssy small sentences for parser detection

import glob
import re

lassy_sents = []

file_list = glob.glob('../Tests/Suites/*.sents')

for file in file_list:
    with open(f'{file}', encoding='utf-8') as infile:
        for line in infile.readlines():
            lassy_sents.append(re.sub(r'.*\|', '', line.strip()))

with open('Data/lassy_sents.txt', 'w', encoding='utf-8') as outfile:
    for sent in lassy_sents:
        outfile.write(sent + '\n')
