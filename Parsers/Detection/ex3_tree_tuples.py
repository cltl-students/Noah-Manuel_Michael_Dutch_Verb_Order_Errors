# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 03.06.2023
# Extract tuples from parses as features for classifiers

import re
import pandas as pd


def get_tuples():
    test_C = pd.read_csv('Data/test_C.tsv', encoding='utf-8', header=0, sep='\t')

    tuple_lists = []

    for tree in test_C['simple_tree']:
        token_tuples = []
        tree = re.sub(r'[(ROOT)\(\)]', '', tree)
        for i, token in enumerate(tree.split()):
            token_tuples.append((i, token))
        tuple_lists.append(token_tuples)

    print(tuple_lists)


if __name__ == '__main__':
    get_tuples()
