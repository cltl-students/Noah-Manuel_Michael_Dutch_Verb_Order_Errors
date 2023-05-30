# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Format sentences to all have the same spacing between punctuation characters through tokenizing with spacy because
# spacy's tokenizer does very good so the input can be different but the output will be the same

import pandas as pd
import spacy


def get_spacy_spaces(spacy_pipeline, df):
    """
    Generate sentences where the empty spaces between tokens are all equal, punctuation is spaced.
    :param spacy_pipeline: preloaded spacy pipeline
    :param df: dataframe with sentences to convert
    :return: df: input dataframe additionally containing converted sentences
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
    Get all spaced sentences for all splits using the get_spacy_spaces() function.
    :return: None
    """
    nlp = spacy.load('nl_core_news_lg')

    for split in ['test', 'dev', 'train']:
        df = pd.read_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', header=0)

        spaced_df = get_spacy_spaces(nlp, df)

        spaced_df.to_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    get_spaced_sentences()
