# Noah-Manuel Michael
# 30.04.2023
# Last updated: 12.05.2023
# Scramble the readability data randomly

import pandas as pd
import re
import spacy
import random


def scramble_readability_data():
    """

    :return:
    """
    # read in the original texts from the readability corpus
    df_texts = pd.read_csv('Data/readability_corpus_edia_preprocessed.tsv', sep='\t', header=0, encoding='utf-8')

    # instantiate pipeline
    nlp = spacy.load('nl_core_news_sm')

    # retrieve a list of all sentences that are longer than 10 characters (otherwise mostly single words, not
    # interesting for word order
    list_of_all_sentences = []
    list_of_all_levels = []
    for i, row in df_texts.iterrows():
        doc = nlp(row['text'])
        for sent in doc.sents:
            if len(sent.text) > 10 and re.match(r'^[A-Z].*[.!?]$', sent.text):
                list_of_all_sentences.append(sent.text)
                list_of_all_levels.append(row['level'])
    print('All texts split into sentences.')

    # save the original sentences in a dataframe before starting to shuffle them, also save the level
    df_orig = pd.DataFrame(list_of_all_sentences, columns=['original'])
    df_orig['level'] = list_of_all_levels

    # tokenize sentence, scramble randomly
    sentences_for_scramble = df_orig['original'].tolist()

    # define lists to store the scrambled sentences in
    all_sentences_tokenized_scrambled = []
    all_sentences_tokenized_scrambled_punctuation_final = []
    all_sentences_tokenized_scrambled_lower = []
    all_sentences_tokenized_scrambled_punctuation_final_lower = []
    all_sentences_tokenized_scrambled_no_punc = []
    all_sentences_tokenized_scrambled_no_punc_lower = []

    for sent in sentences_for_scramble:
        # add empty spaces before and after commas, so they are recognized as tokens of their own, get rid of tabs
        if ', ' in sent:
            sent = re.sub(r', ', ' , ', sent)

        # check for punctuation marks at the end of sentences, tokenize the sentences, append shuffled sentences to list

        # dot
        if sent.endswith('.'):
            sent = re.sub(r'\.', ' .', sent).split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])  # randomly shuffled
            all_sentences_tokenized_scrambled_lower.append([tok.lower() for tok in sent])  # randomly shuffled,
            # lowercase
            sent.remove('.')
            all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['.'])  # randomly shuffled, punctuation
            # at the end is preserved
            all_sentences_tokenized_scrambled_punctuation_final_lower.append([tok.lower() for tok in sent] + ['.'])
            # randomly shuffled, punctuation at the end is preserved, lowercase

        # question mark
        elif sent.endswith('?'):
            sent = re.sub(r'\?', ' ?', sent).split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])
            all_sentences_tokenized_scrambled_lower.append([tok.lower() for tok in sent])
            sent.remove('?')
            all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['?'])
            all_sentences_tokenized_scrambled_punctuation_final_lower.append([tok.lower() for tok in sent] + ['?'])

        # exclamation mark
        elif sent.endswith('!'):
            sent = re.sub(r'!', ' !', sent).split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])
            all_sentences_tokenized_scrambled_lower.append([tok.lower() for tok in sent])
            sent.remove('!')
            all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['!'])
            all_sentences_tokenized_scrambled_punctuation_final_lower.append([tok.lower() for tok in sent] + ['!'])

        # no final punctuation
        else:
            sent = sent.split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])
            all_sentences_tokenized_scrambled_lower.append([tok.lower() for tok in sent])
            all_sentences_tokenized_scrambled_punctuation_final.append(sent)
            all_sentences_tokenized_scrambled_punctuation_final_lower.append([tok.lower() for tok in sent])

        # remove all punctuation and save the shuffled sentences
        all_sentences_tokenized_scrambled_no_punc.append([tok for tok in sent if tok not in [',', '.', '!', '?', ';']])
        all_sentences_tokenized_scrambled_no_punc_lower.append([tok.lower() for tok in sent if tok not in
                                                                [',', '.', '!', '?', ';']])

    # save all scrambled sentences in a df
    df_orig['scrambled'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled]
    df_orig['scrambled_lower'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_lower]
    df_orig['scrambled_final_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_punctuation_final]
    df_orig['scrambled_final_punc_lower'] = [' '.join(sent) for sent in
                                            all_sentences_tokenized_scrambled_punctuation_final_lower]
    df_orig['scrambled_no_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_no_punc]
    df_orig['scrambled_no_punc_lower'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_no_punc_lower]

    # write the df to file
    df_orig.to_csv('readability_data_scrambled.tsv', sep='\t', header=True, encoding='utf-8', index_label='index')
    print('Scrambled data saved to file.')


if __name__ == '__main__':
    scramble_readability_data()
