# Noah-Manuel Michael
# Created: 09.06.2023
# Last updated: 09.06.2023
# Get a random sample of 1 million train and 50000 dev sents for finetuning

import pandas as pd


def get_random_sample_for_fine_tuning():
    df_train = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/train_shuffled_random_all_and_verbs.tsv',
                     sep='\t', encoding='utf-8', header=0)

    df_train_sampled = df_train.sample(n=1000000, random_state=1)
    df_train_sampled.reset_index(drop=True, inplace=True)

    df_train_sampled.to_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                      'train_shuffled_random_all_and_verbs_sampled_transformer.tsv', encoding='utf-8', sep='\t',
                      header=True, index_label='index_new')

    df_dev = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/dev_shuffled_random_all_and_verbs.tsv',
                     sep='\t', encoding='utf-8', header=0)

    df_dev_sampled = df_dev.sample(n=50000, random_state=1)
    df_dev_sampled.reset_index(drop=True, inplace=True)

    df_dev_sampled.to_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                      'dev_shuffled_random_all_and_verbs_sampled_transformer.tsv', encoding='utf-8', sep='\t',
                      header=True, index_label='index_new')


if __name__ == '__main__':
    get_random_sample_for_fine_tuning()
