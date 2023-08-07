# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 09.06.2023
# Extract tuples from parses as features for classifiers with spaCy

import pandas as pd
import spacy
import json
import argparse


def get_tuples_with_spacy(split, dataset):
    """
    Get tuples of position-PoS from spaCy.
    :param str dataset: no_punc, scrambled_no_punc, verbs_random_no_punc
    :return:
    """
    short_name_mapping = {'no_punc': 'Correct', 'scrambled_no_punc': 'Rand', 'verbs_random_no_punc': 'Verbs',
                          'tendencies_no_punc': 'Info'}
    short_name = short_name_mapping[dataset]

    if not short_name == 'Info':
        df = pd.read_csv(f'../../Data/Dataset_Construction/Permuted_Datasets/{split}_shuffled_random_all_and_verbs.tsv',
                            encoding='utf-8', header=0, sep='\t')
    else:
        df = pd.read_csv(f'../../Data/Dataset_Construction/Permuted_Datasets/'
                         f'{split}_shuffled_random_all_and_verbs_and_tendencies.tsv', encoding='utf-8', header=0,
                         sep='\t')

    nlp = spacy.load('nl_core_news_lg')

    for i, row in df.iterrows():
        token_tuples = []
        doc = nlp(row[dataset])
        for position, token in enumerate(doc):
            token_tuples.append((position, token.pos_))
        with open(f'Data/Tuples/Spacy/tuple_data_spacy_{split}_{short_name}.json', 'a') as outfile:
            json.dump({dataset: token_tuples}, outfile)
            outfile.write('\n')


def get_learner_tuples_with_spacy():
    """
    Get tuples of position-PoS from spaCy for learner sentences.
    :return:
    """
    sent_list = []

    with open('../../Data/Annotation/Data/leerder_sents_no_punc_for_testing.tsv') as infile:
        for line in infile.readlines():
            sent_list.append(line.strip())

    nlp = spacy.load('nl_core_news_lg')

    for i, sent in enumerate(sent_list):
        token_tuples = []
        doc = nlp(sent)
        for position, token in enumerate(doc):
            token_tuples.append((position, token.pos_))
        with open(f'Data/Tuples/Spacy/tuple_data_spacy_test_Learn.json', 'a') as outfile:
            json.dump({'no_punc': token_tuples}, outfile)
            outfile.write('\n')


if __name__ == '__main__':

    for split in ['train', 'dev', 'test']:
        for dataset in ['tendencies_no_punc', 'no_punc', 'scrambled_no_punc', 'verbs_random_no_punc']:
            get_tuples_with_spacy(split, dataset)

    get_learner_tuples_with_spacy()
