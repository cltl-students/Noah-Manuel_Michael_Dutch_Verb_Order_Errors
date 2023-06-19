# Noah-Manuel Michael
# Created: 19.06.2023
# Last updated: 19.06.2023
# Analyze phrase structure

import pandas as pd
from collections import Counter


def analyze_phrase_structure():
    """

    :return:
    """
    df_main = pd.read_csv('finite_main_errors.tsv', sep='\t', encoding='utf-8', header=None)
    df_sub = pd.read_csv('finite_sub_errors.tsv', sep='\t', encoding='utf-8', header=None)

    print('Phrase structures main clauses:')
    print(Counter(df_main[1]))
    print('Phrase structures subclauses:')
    print(Counter(df_sub[1]))


if __name__ == '__main__':
    analyze_phrase_structure()
