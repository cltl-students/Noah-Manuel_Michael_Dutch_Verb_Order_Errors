# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 06.06.2023
# Extract tuples from parses as features for classifiers with spaCy
# The function was simplified with ChatGPT to allow for more efficient running on SURF (for each dataset in parallel)

import pandas as pd
import spacy
import json
import argparse


def get_tuples_with_spacy(dataset):
    for split in ['train']:  # , 'dev', 'test']:
        train = pd.read_csv(f'{split}_shuffled_random_all_and_verbs.tsv',
                            encoding='utf-8', header=0, sep='\t')

        nlp = spacy.load('nl_core_news_lg')

        for i, row in train.iterrows():
            token_tuples = []
            try:
                doc = nlp(row[dataset])
                for position, token in enumerate(doc):
                    token_tuples.append((position, token.pos_))
                with open(f'tuple_data_{split}_spacy_{dataset}.json', 'a') as outfile:
                    json.dump({dataset: token_tuples}, outfile)
                    outfile.write('\n')
            except ValueError:
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='Dataset to use')

    args = parser.parse_args()

    get_tuples_with_spacy(args.dataset)


# def get_tuples():
#     train = pd.read_csv('train_shuffled_random_all_and_verbs_old.tsv',
#                         encoding='utf-8', header=0, sep='\t')
#
#     nlp = spacy.load('nl_core_news_lg')
#
#     for i, row in train[:10].iterrows():
#         token_tuples_c = []
#         token_tuples_ar = []
#         token_tuples_vr = []
#
#         doc_c = nlp(row['spaced'])
#         doc_ar = nlp(row['scrambled_final_punc'])
#         doc_vr = nlp(row['verbs_random_punc_final'])
#
#         for position, token in enumerate(doc_c):
#             token_tuples_c.append((position, token.text))
#
#         for position, token in enumerate(doc_ar):
#             token_tuples_ar.append((position, token.text))
#
#         for position, token in enumerate(doc_vr):
#             token_tuples_vr.append((position, token.text))
#
#         with open('tuple_data_train_spacy.json', 'a') as outfile:
#             json.dump({'spaced': token_tuples_c, 'scrambled_final_punc': token_tuples_ar,
#                        'verbs_random_punc_final': token_tuples_vr}, outfile)
#
#
# if __name__ == '__main__':
#     get_tuples()
