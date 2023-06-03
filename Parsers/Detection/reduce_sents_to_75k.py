# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 03.06.2023
# Reduce the parsed dev data to 75000 sents

import pandas as pd


def reduce_to_75k():
    df_dev_correct = pd.read_csv('Data/dev_correct_parsed.tsv', sep='\t', header=0, encoding='utf-8')
    df_dev_incorrect = pd.read_csv('Data/dev_incorrect_parsed.tsv', sep='\t', header=0, encoding='utf-8')

    df_dev_correct[:75000].to_csv('Data/dev_correct_parsed.tsv', sep='\t', index=False, encoding='utf-8')
    df_dev_incorrect[:75000].to_csv('Data/dev_incorrect_parsed.tsv', sep='\t', index=False, encoding='utf-8')


if __name__ == '__main__':
    reduce_to_75k()
