# Noah-Manuel Michael
# Created: 26.05.2023
# Last updated: 26.05.2023
# Get rid of sentences longer than 50 tokens to save computational power and imitate the original learner data

import pandas as pd
import spacy


def get_filter_sents_for_length(df, spacy_pipeline):
    """

    :param df:
    :param spacy_pipeline:
    :return:
    """
    nlp = spacy_pipeline
    filter_sents_up_to_50 = []

    for sent in df['original']:
        doc = nlp(sent)
        if len(doc) <= 50:
            filter_sents_up_to_50.append(True)
        else:
            filter_sents_up_to_50.append(False)

    return filter_sents_up_to_50


def filter_datasplit_for_sentence_length():
    """

    :return:
    """
    nlp = spacy.load('nl_core_news_lg')

    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', header=0)

        filter_sents_up_to_50 = get_filter_sents_for_length(df, nlp)

        df_filtered = df.loc[filter_sents_up_to_50]
        df_filtered.to_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    filter_datasplit_for_sentence_length()
