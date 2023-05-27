# Noah-Manuel Michael
# Created: 25.05.2023
# Last updated: 25.05.2023
# Preprocess the Leipzig corpora

import re
import pandas as pd


def preprocess_leipzig_corpora():
    """

    :return:
    """
    df_mixed = pd.read_csv('../Unpermuted_Datasets/Originals/leipzig_corpus/nld_mixed_2012_1M-sentences.txt',
                           sep='\t', encoding='utf-8', names=['index', 'sent'])
    df_news = pd.read_csv('../Unpermuted_Datasets/Originals/leipzig_corpus/nld_news_2022_1M-sentences.txt',
                          sep='\t', encoding='utf-8', names=['index', 'sent'])
    df_wiki = pd.read_csv('../Unpermuted_Datasets/Originals/leipzig_corpus/nld_wikipedia_2021_1M-sentences.txt',
                          sep='\t', encoding='utf-8', names=['index', 'sent'])

    # make sure there are no duplicate sentences in the corpora by taking the set
    all_sents = {re.sub(r'[\t ]', ' ', sent) for sent
                 in (df_wiki['sent'].tolist() + df_mixed['sent'].tolist() + df_news['sent'].tolist())
                 if len(sent) > 10 and re.match(r'^[A-Z].*[.!?]$', sent)}

    with open('../Unpermuted_Datasets/Originals/leipzig_data.tsv', 'w', encoding='utf-8') as outfile:
        outfile.write('index\toriginal\n')
        for i, sent in enumerate(all_sents):
            outfile.write(f'{i}\t{sent}\n')


if __name__ == '__main__':
    preprocess_leipzig_corpora()
