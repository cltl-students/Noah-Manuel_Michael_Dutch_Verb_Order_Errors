# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Unite the preprocessed datasets and split into train, dev, and test datasets; only include sentences with at least one
# verb token according to spaCy

import pandas as pd
import spacy


def check_if_verb_in_sent(spacy_pipeline, sents):
    """
    Check whether sentences have at least one verb. Return the sentences with at least one verb.
    :param spacy_pipeline: the instantiated spaCy pipeline
    :param list sents: a list of sentences to parse
    :return: verb_sents: a list of sentences containing at least one verb according to spaCy
    """
    nlp = spacy_pipeline
    verb_sents = []
    for sent in sents:
        verb_in_sent = False
        doc = nlp(sent)
        for token in doc:
            if token.pos_ in ['VERB', 'AUX']:
                verb_in_sent = True
        if verb_in_sent:
            verb_sents.append(sent)

    return verb_sents


def get_datasplit():
    """
    Read in all the datasets. Make test dataset from edia data only - aiming to approximate the levels seen in the
    actual learner data. Use the sentences that are outside the learner level range for train and dev. Combine the
    leftover sents plus all other datasets. Get a number of sentence % 10 == 0 for dev vs number of
    sentence % 10 != 0 for train split.
    :return:
    """
    df_edia = pd.read_csv('readability_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_lassy = pd.read_csv('lassy_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_wainot = pd.read_csv('wainot_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_leipzig = pd.read_csv('leipzig_data.tsv', sep='\t', encoding='utf-8', header=0)

    nlp = spacy.load('nl_core_news_lg')

    # reduce the test data to the levels we see in the actual learner data
    levels_for_test_set = ['A2', 'A2+', 'B1', 'B1+', 'B2', 'B2+', 'C1']
    df_levels_filtered = df_edia.loc[[True if level in levels_for_test_set else False for level in df_edia['level']]]

    all_test_sents_with_verbs = []
    all_test_sents_with_verbs_levels = []

    # make sure there is at least one verb in the sentence
    for i, row in df_levels_filtered.iterrows():
        verb_in_sent = False
        doc = nlp(row['original'])
        for token in doc:
            if token.pos_ in ['VERB', 'AUX']:
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

    leftover_sents_with_verbs = check_if_verb_in_sent(nlp, leftover_sents)
    print('Leftover Edia data checked for verbs.')
    lassy_sents_with_verbs = check_if_verb_in_sent(nlp, df_lassy['original'].tolist())
    print('Lassy data checked for verbs.')
    wainot_sents_with_verbs = check_if_verb_in_sent(nlp, df_wainot['original'].tolist())
    print('Wainot data checked for verbs.')
    leipzig_sents_with_verbs = check_if_verb_in_sent(nlp, df_leipzig['original'].tolist())
    print('Leipzig data checked for verbs.')

    # combine all other data
    combined_data = lassy_sents_with_verbs + wainot_sents_with_verbs + leftover_sents_with_verbs + \
                    leipzig_sents_with_verbs

    # get every 10th sentence for dev data
    df_dev = pd.DataFrame([sent for i, sent in enumerate(combined_data) if i % 10 == 0], columns=['original'])

    # all other sents for train data
    df_train = pd.DataFrame([sent for i, sent in enumerate(combined_data) if i % 10 != 0], columns=['original'])

    df_test.to_csv('Unpermuted_Datasets/test.tsv', sep='\t', index_label='index', encoding='utf-8')
    df_dev.to_csv('Unpermuted_Datasets/dev.tsv', sep='\t', index_label='index', encoding='utf-8')
    df_train.to_csv('Unpermuted_Datasets/train.tsv', sep='\t', index_label='index', encoding='utf-8')


if __name__ == '__main__':
    get_datasplit()
