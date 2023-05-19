# Noah-Manuel Michael
# Created: 19.05.2023
# Last updated: 19.05.2023
# Experiments with decorators

from itertools import permutations

permutations_2tokvp_adjacent = permutations(['Ik', 'een', 'boek', 'wil', 'kopen'])


for i, permutation in enumerate(permutations_2tokvp_adjacent):
    print(i, permutation)
