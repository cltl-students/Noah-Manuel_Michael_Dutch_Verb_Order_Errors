# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 24.05.2023
# Unite the preprocessed datasets and split into train, dev, and test datasets; only include sentences with at least one
# verb token according to frog parser

import re
import pandas as pd
from frog import Frog, FrogOptions


def check_if_verb_in_sent(frog_parser, sents):
    """
    Check whether sentences have at least one verb. Return the sentences with at least one verb.
    :param frog_parser: the instantiated frog parser
    :param list sents: a list of sentences to parse
    :return: verb_sents: a list of sentences containing at least one verb according to the frog parser
    """
    frog = frog_parser
    verb_sents = []
    for sent in sents:
        if len(sent.split()) <= 10:  # reduce max sentence length to 30, otherwise parsing takes too much memory
            verb_in_sent = False
            output = frog.process(sent)
            for token in output:
                if re.findall(r'.-VP', token['chunker']):
                    verb_in_sent = True
            if verb_in_sent:
                verb_sents.append(sent)

    return verb_sents


def get_datasplit():
    """

    :return:
    """
    df_edia = pd.read_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset_Construction/Unpermuted '
                          'Datasets/readability_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_lassy = pd.read_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset_Construction/Unpermuted '
                           'Datasets/lassy_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_wainot = pd.read_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset_Construction/Unpermuted '
                            'Datasets/wainot_data.tsv', sep='\t', encoding='utf-8', header=0)

    frog = Frog(FrogOptions(parser=True))

    # reduce the test data to the levels we see in the actual learner data
    levels_for_test_set = ['A2', 'A2+', 'B1', 'B1+', 'B2', 'B2+', 'C1']
    df_levels_filtered = df_edia.loc[[True if level in levels_for_test_set else False for level in df_edia['level']]]

    all_test_sents_with_verbs = []
    all_test_sents_with_verbs_levels = []

    # make sure there is at least one verb in the sentence
    for i, row in df_levels_filtered.iterrows():
        if len(row['original'].split()) <= 10:
            verb_in_sent = False
            output = frog.process(row['original'])
            for token in output:
                if re.findall(r'.-VP', token['chunker']):
                    verb_in_sent = True
            if verb_in_sent:
                all_test_sents_with_verbs.append(row['original'])
                all_test_sents_with_verbs_levels.append(row['level'])
    print('Test data checked for verbs.')

    df_test = pd.DataFrame({'original': all_test_sents_with_verbs,  # make new df to get correct indices
                            'level': all_test_sents_with_verbs_levels})

    leftover_sents = []  # use the sents that are outside the level range for train and dev data
    for i, row in df_edia.iterrows():
        if row['level'] not in levels_for_test_set:
            leftover_sents.append(row['original'])

    leftover_sents_with_verbs = check_if_verb_in_sent(frog, leftover_sents)
    print('Leftover data checked for verbs.')
    lassy_sents_with_verbs = check_if_verb_in_sent(frog, df_lassy['original'].tolist())
    print('Lassy data checked for verbs.')
    wainot_sents_with_verbs = check_if_verb_in_sent(frog, df_wainot['original'].tolist())
    print('Wainot data checked for verbs.')

    # combine all other data
    combined_data = lassy_sents_with_verbs + wainot_sents_with_verbs + leftover_sents_with_verbs

    # get every 10th sentence for dev data
    df_dev = pd.DataFrame([sent for i, sent in enumerate(combined_data) if i % 10 == 0], columns=['original'])

    # all other sents for train data
    df_train = pd.DataFrame([sent for i, sent in enumerate(combined_data) if i % 10 != 0], columns=['original'])

    df_test.to_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset_Construction/Unpermuted_Datasets/'
                   'test_shuffled_random_all.tsv', sep='\t', index_label='index', encoding='utf-8')
    df_dev.to_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset_Construction/Unpermuted_Datasets/'
                  'dev_shuffled_random_all.tsv', sep='\t', index_label='index', encoding='utf-8')
    df_train.to_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset_Construction/Unpermuted_Datasets/'
                    'train_shuffled_random_all.tsv', sep='\t', index_label='index', encoding='utf-8')


if __name__ == '__main__':
    get_datasplit()
