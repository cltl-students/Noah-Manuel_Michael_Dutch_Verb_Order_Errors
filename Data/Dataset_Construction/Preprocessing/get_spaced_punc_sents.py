# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Format sentences to have spaced punctuation in line with Lassy sentences

import pandas as pd
import spacy


def get_spacy_spaces(spacy_pipeline, df):
    """

    :param spacy_pipeline:
    :param df:
    :return:
    """
    nlp = spacy_pipeline
    spaced_sents = []

    for sent in df['original']:
        spaced_sent = []
        doc = nlp(sent)
        for token in doc:
            spaced_sent.append(token.text)
        spaced_sents.append(' '.join(spaced_sent))

    df['spaced'] = spaced_sents

    return df


def get_spaced_sentences():
    """

    :return:
    """
    nlp = spacy.load('nl_core_news_lg')

    for split in ['test', 'dev', 'train']:
        df = pd.read_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', header=0)

        spaced_df = get_spacy_spaces(nlp, df)

        spaced_df.to_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    get_spaced_sentences()
