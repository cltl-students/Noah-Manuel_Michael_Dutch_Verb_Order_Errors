# Noah-Manuel Michael
# Created: 19.05.2023
# Last updated: 19.05.2023
# Experiments for only permuting the verb positions

from itertools import permutations

sentence = ['Ik', 'wil', 'een', 'boek', 'kopen']
sentence_only_verbs = ['0', 'wil', '0', '0', 'kopen']  # get only the verb tokens

# get all permutations
set_sentence = set()
for permutation in permutations(sentence):  # get permutations
    set_sentence.add(permutation)  # 120 possible

# get permutations of only the verb tokens (indifferent of B- I-)
set_sentence_only_verbs = set()
for permutation in permutations(sentence_only_verbs):  # get permutations
    set_sentence_only_verbs.add(permutation)  # reduce to unique ones, 20 possible

for permutation in set_sentence_only_verbs:
    cache = [el for el in sentence if el not in sentence_only_verbs]  # get all words that are not verbs
    permutation = list(permutation)  # turn tuple to list
    for i, element in enumerate(permutation):  # go through list and replace any '0' with leftmost token in cache,
        # keeping the relative order between non-verb tokens
        if element == '0':
            permutation[i] = cache.pop(0)
    print(permutation)

