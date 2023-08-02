from itertools import permutations

masked_sent = [0, 'wil', 0, 0, 'kopen']
cache = ['ik', 'geen', 'boeken']

for p in set(permutations(masked_sent)):
    print(p)
