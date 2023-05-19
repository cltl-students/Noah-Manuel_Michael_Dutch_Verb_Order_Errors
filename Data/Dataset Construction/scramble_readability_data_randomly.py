# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 18.05.2023
# Scramble the readability data randomly

import pandas as pd
import re
import spacy
import random


def scramble_readability_data():
    """
    Read in the preprocessed readability corpus data (texts). Split the texts into sentences. Save different versions of
    the sentences in a new file.
    Sentence types:
    original - original with spaced punctuation - original with no punctuation - scrambled - scrambled with final
    punctuation - scrambled with no punctuation
    :return: None
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
            if len(sent.text) > 10 and re.match(r'^[A-Z].*[.!?]$', sent.text):  # only sents that start with capital
                # letter and end if with [.?!] -> complete sentences, no bullet points, etc.
                list_of_all_sentences.append(sent.text)
                list_of_all_levels.append(row['level'])
    print('All texts split into sentences.')

    # save the original sentences in a dataframe before starting to shuffle them, also save the level
    df_orig = pd.DataFrame(list_of_all_sentences, columns=['original'])
    df_orig['level'] = list_of_all_levels

    # save the original sentence but insert a space before punctuation tokens in order to ensure uniform tokenization by
    # the transformer tokenizers
    orig_sentences_spaced_punc = []

    for sent in df_orig['original'].tolist():
        if ', ' in sent:
            sent = re.sub(r', ', ' , ', sent)
        if sent.endswith('.'):
            orig_sentences_spaced_punc.append(re.sub(r'\.$', ' .', sent))
        elif sent.endswith('?'):
            orig_sentences_spaced_punc.append(re.sub(r'\?$', ' ?', sent))
        elif sent.endswith('!'):
            orig_sentences_spaced_punc.append(re.sub(r'!$', ' !', sent))
        else:
            orig_sentences_spaced_punc.append(sent)

    df_orig['original_spaced_punc'] = orig_sentences_spaced_punc

    # save the original sentence with no punctuation
    orig_sentences_no_punc = []
    for sent in df_orig['original'].tolist():
        orig_sentences_no_punc.append(re.sub(r'[.?!,;]', '', sent))
    df_orig['original_no_punc'] = orig_sentences_no_punc

    # define lists to store the scrambled sentences in
    all_sentences_tokenized_scrambled = []
    all_sentences_tokenized_scrambled_punctuation_final = []
    all_sentences_tokenized_scrambled_no_punc = []

    for sent in df_orig['original'].tolist():
        # add empty spaces before and after commas, so they are recognized as tokens of their own
        if ', ' in sent:
            sent = re.sub(r', ', ' , ', sent)
        # check for punctuation marks at the end of sentences, tokenize the sentences, append shuffled sentences to list
        if sent.endswith('.'):  # full stop
            sent = re.sub(r'\.', ' .', sent).split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])  # randomly shuffled
            sent.remove('.')
            all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['.'])  # randomly shuffled, punctuation
            # at the end is preserved
        elif sent.endswith('?'):  # question mark
            sent = re.sub(r'\?', ' ?', sent).split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])
            sent.remove('?')
            all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['?'])
        elif sent.endswith('!'):
            sent = re.sub(r'!', ' !', sent).split()
            random.shuffle(sent)
            all_sentences_tokenized_scrambled.append([tok for tok in sent])
            sent.remove('!')
            all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['!'])

        # remove all punctuation and save the shuffled sentences
        all_sentences_tokenized_scrambled_no_punc.append([tok for tok in sent if tok not in [',', '.', '!', '?', ';']])

    # save all scrambled sentences in a df
    df_orig['scrambled'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled]
    df_orig['scrambled_final_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_punctuation_final]
    df_orig['scrambled_no_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_no_punc]

    # write the df to file
    df_orig.to_csv('Data/readability_data_scrambled.tsv', sep='\t', header=True, encoding='utf-8', index_label='index')
    print('Scrambled data saved to file.')


if __name__ == '__main__':
    scramble_readability_data()
