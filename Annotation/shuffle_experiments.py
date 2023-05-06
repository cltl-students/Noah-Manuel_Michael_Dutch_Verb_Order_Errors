# Noah-Manuel Michael
# 06.05.2023
# Shuffle sentences and get all possible permutations

import random
import itertools
import re

with open('Data/short_story.txt') as infile:
    sentences = []
    for line in infile.readlines():
        print(line)
        if line.endswith('.\n'):
            normalized_line = re.sub(r'\.\n', ' .', line)
        elif line.endswith('?\n'):
            normalized_line = re.sub(r'\?\n', ' ?', line)
        elif line.endswith('!\n'):
            normalized_line = re.sub(r'!\n', ' !', line)
        sentences.append(normalized_line.split())

print(sentences)

# for rescoring and word order correction
permutations = itertools.permutations(sentences[-1])
# print(len([p for p in permutations]))
for p in permutations:
    print(p)
    list_of_p = list(p)
    list_of_p.remove('')
    list_of_p.extend('.')
    permuted_sentence = ' '.join(list_of_p)
    print(permuted_sentence.split())

# for dataset creation
# for sent in sentences:
#     random.shuffle(sent)
#
# print(sentences)
