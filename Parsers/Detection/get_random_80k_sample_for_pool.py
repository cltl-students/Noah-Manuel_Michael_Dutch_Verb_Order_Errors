# Noah-Manuel Michael
# Created: 09.06.2023
# Last updated: 09.06.2023
# Get a random sample of 80000 sents for the pool experiments (train)

import pandas as pd


def get_random_80k_sample():
    df = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/train_shuffled_random_all_and_verbs.tsv',
                     sep='\t', encoding='utf-8', header=0)

    df_sampled = df.sample(n=80000, random_state=1)
    df_sampled.reset_index(drop=True, inplace=True)

    df_sampled.to_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                      'train_shuffled_random_all_and_verbs_sampled.tsv', encoding='utf-8', sep='\t', header=True,
                      index_label='index_new')


if __name__ == '__main__':
    get_random_80k_sample()
