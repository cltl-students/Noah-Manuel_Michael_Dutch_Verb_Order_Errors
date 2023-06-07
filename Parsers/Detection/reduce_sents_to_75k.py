# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 03.06.2023
# Reduce the parsed dev data to 75000 sents

import pandas as pd


def reduce_to_75k():
    # df_dev_correct = pd.read_csv('Data/dev_C.tsv', sep='\t', header=0, encoding='utf-8')
    # df_dev_incorrect = pd.read_csv('Data/dev_AR.tsv', sep='\t', header=0, encoding='utf-8')
    df_dev_incorrect_verbs = pd.read_csv('Data/Trees/dev_VR.tsv', sep='\t', header=0, encoding='utf-8')

    # df_dev_correct[:75000].to_csv('Data/dev_C.tsv', sep='\t', index=False, encoding='utf-8')
    # df_dev_incorrect[:75000].to_csv('Data/dev_AR.tsv', sep='\t', index=False, encoding='utf-8')
    df_dev_incorrect_verbs[:75000].to_csv('Data/dev_VR.tsv', sep='\t', index=False, encoding='utf-8')


if __name__ == '__main__':
    reduce_to_75k()
