# Noah-Manuel Michael
# Created: 31.05.2023
# Last updated: 11.06.2023
# Inspect to what extent simplifying the tree structures renders fewer unique trees

import pandas as pd
from collections import Counter


def get_num_of_unique_trees():
    """

    :return:
    """
    pool = pd.read_csv('Data/Trees/train_Correct.tsv', sep='\t', header=0, encoding='utf-8')
    test_C = pd.read_csv('Data/Trees/test_Correct.tsv', sep='\t', header=0, encoding='utf-8')
    test_AR = pd.read_csv('Data/Trees/test_Rand.tsv', sep='\t', header=0, encoding='utf-8')
    test_VR = pd.read_csv('Data/Trees/test_Verbs.tsv', sep='\t', header=0, encoding='utf-8')

    for df in [pool, test_C, test_AR, test_VR]:
        print('___')
        print('Num trees in original parses:')
        print(len(Counter(df['tree'])))

        print('Num trees in simplified parses:')
        print(len(Counter(df['simple_tree'])))


if __name__ == '__main__':
    get_num_of_unique_trees()
