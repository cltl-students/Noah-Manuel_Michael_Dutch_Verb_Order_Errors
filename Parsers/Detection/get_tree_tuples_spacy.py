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
    short_name_mapping = {'no_punc': 'C', 'scrambled_no_punc': 'AR', 'verbs_random_no_punc': 'VR'}
    short_name = short_name_mapping[dataset]

    df = pd.read_csv(f'../Permuted_Datasets/{split}_shuffled_random_all_and_verbs.tsv',
                        encoding='utf-8', header=0, sep='\t')

    nlp = spacy.load('nl_core_news_lg')

    for i, row in df.iterrows():
        token_tuples = []
        doc = nlp(row[dataset])
        for position, token in enumerate(doc):
            token_tuples.append((position, token.pos_))
        with open(f'tuple_data_spacy_{split}_{short_name}.json', 'a') as outfile:
            json.dump({dataset: token_tuples}, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    for split in ['train', 'dev', 'test']:
        for dataset in ['no_punc', 'scrambled_no_punc', 'verbs_random_no_punc']:
            get_tuples_with_spacy(split, dataset)
