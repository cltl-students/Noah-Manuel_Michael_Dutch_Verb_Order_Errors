# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 23.05.2023
# Preprocess the readability data

import pandas as pd
import re
import spacy
from collections import defaultdict, Counter


def preprocess_readability_corpus():
    """
    Read in the readability corpus by Edia, average the CEFR levels for each text, split the texts into sentences and
    write the sentences + their corresponding CEFR levels to file.
    :return: None
    """
    # read in the original dataset
    df = pd.read_csv('../Unpermuted Datasets/readability_corpus_edia_original.csv', sep=',', header=0, encoding='utf-8')

    # get rid of new line characters and double spaces
    all_texts = [re.sub(r'\n', ' ', text) for text in df['text_plain']]
    all_texts = [re.sub(r'  +', ' ', text) for text in all_texts]
    df['text_normalized'] = all_texts

    # check which texts appear less or more than 3 times
    text_to_amount_of_annotators = defaultdict(int)
    for key, value in Counter(df['text_normalized']).items():
        if value != 3:
            text_to_amount_of_annotators[key] = value

    # map levels to points and vice versa
    level_to_points = {'A1': 1, 'A1+': 2, 'A2': 3, 'A2+': 4, 'B1': 5, 'B1+': 6, 'B2': 7, 'B2+': 8, 'C1': 9, 'C1+': 10,
                       'C2': 11, 'C2+': 12}
    points_to_level = {value: key for key, value in level_to_points.items()}

    # get the points for each text
    level_points = []
    for level in df['cefr_level']:
        level_points.append(level_to_points[level])

    # add the points to the dataframe
    df['level_points'] = level_points

    # get a mapping of texts and sum of points
    texts_to_sum = defaultdict(int)
    for i, row in df.iterrows():
        texts_to_sum[row['text_normalized']] += row['level_points']

    # calculate the average for each text and map back to level
    texts_to_average_level = defaultdict(str)
    for text, sum_of_level_points in texts_to_sum.items():
        if text not in text_to_amount_of_annotators.keys():
            texts_to_average_level[text] = points_to_level[round(sum_of_level_points / 3)]
        else:  # get the actual amount of annotators if they are not 3
            texts_to_average_level[text] = points_to_level[round(sum_of_level_points /
                                                                 text_to_amount_of_annotators[text])]

    df_texts = pd.DataFrame([text for text in texts_to_average_level.keys()], columns=['text'])
    df_texts['level'] = [level for level in texts_to_average_level.values()]
    print('CEFR level for each text averaged.')

    # instantiate pipeline
    nlp = spacy.load('nl_core_news_sm')

    # retrieve a list of all sentences that are longer than 10 characters (otherwise mostly single words, not
    # interesting for word order
    list_of_all_sentences = []
    list_of_all_levels = []
    for i, row in df_texts.iterrows():
        doc = nlp(row['text'])
        for sent in doc.sents:
            if len(sent.text) > 10 and re.match(r'^[A-Z].*[.!?]$', sent.text):  # only sents that start with capital
                # letter and end if with [.?!] -> complete sentences, no bullet points, etc.
                list_of_all_sentences.append(sent.text)
                list_of_all_levels.append(row['level'])
    print('All texts split into sentences.')

    # save the original sentences in a dataframe before starting to shuffle them, also save the level
    df_sents = pd.DataFrame(list_of_all_sentences, columns=['original'])
    df_sents['level'] = list_of_all_levels

    df_sents.to_csv('Unpermuted Datasets/readability_data.tsv', sep='\t', header=True, encoding='utf-8',
                    index_label='index')
    print('Sents and corresponding levels saved to file.')


if __name__ == '__main__':
    preprocess_readability_corpus()
