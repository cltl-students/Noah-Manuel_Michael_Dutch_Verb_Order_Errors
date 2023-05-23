# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 21.05.2023
# Scramble the readability data randomly

import pandas as pd
import re
import random

# TO-DO: Integrate the preprocessing of punctuation in a different script, also change in l_shuffle_only_verbs
# TO-DO: make sure punctuation is not the only thing that was scrambled since when removing punc, there is no difference
# between the original and the scrambled sentence in that case


def shuffle_sentences_randomly(path_to_file):
    """
    Read in the preprocessed readability corpus data. Get a single random permutation for each sentence. Save different
    versions of the sentences in a new file.
    Sentence types:
    original - original with spaced punctuation - original with no punctuation - scrambled - scrambled with final
    punctuation - scrambled with no punctuation
    :return: None
    """
    # read in the original texts from the readability corpus
    df_sents = pd.read_csv(path_to_file, sep='\t', header=0, encoding='utf-8')

    # save the original sentence but insert a space before punctuation tokens in order to ensure uniform tokenization by
    # the transformer tokenizers
    orig_sentences_spaced_punc = []

    for sent in df_sents['original'].tolist():
        if ', ' in sent:
            sent = re.sub(r', ', ' , ', sent)
        if sent.endswith('.'):
            orig_sentences_spaced_punc.append(re.sub(r'\.$', ' .', sent))
        elif sent.endswith('?'):
            orig_sentences_spaced_punc.append(re.sub(r'\?$', ' ?', sent))
        elif sent.endswith('!'):
            orig_sentences_spaced_punc.append(re.sub(r'!$', ' !', sent))

    df_sents['original_spaced_punc'] = orig_sentences_spaced_punc

    # save the original sentence with no punctuation
    orig_sentences_no_punc = []
    for sent in df_sents['original'].tolist():
        orig_sentences_no_punc.append(re.sub(r'[.?!,;:]', '', sent))
    df_sents['original_no_punc'] = orig_sentences_no_punc

    # define lists to store the scrambled sentences in
    all_sentences_tokenized_scrambled = []
    all_sentences_tokenized_scrambled_punctuation_final = []
    all_sentences_tokenized_scrambled_no_punc = []

    for sent in df_sents['original'].tolist():
        # add empty spaces before and after commas, so they are recognized as tokens of their own
        if ', ' in sent:
            sent = re.sub(r', ', ' , ', sent)
        # check for punctuation marks at the end of sentences, tokenize the sentences, append shuffled sentences to list
        if sent.endswith('.'):  # full stop
            sent = re.sub(r'\.$', ' .', sent).split()
            copy_of_sent = sent.copy()  
            while copy_of_sent == sent:  # make sure the shuffled version != the original version
                random.shuffle(copy_of_sent)
            all_sentences_tokenized_scrambled.append([tok for tok in copy_of_sent])  # randomly shuffled
            copy_of_sent.remove('.')
            all_sentences_tokenized_scrambled_punctuation_final.append(copy_of_sent + ['.'])  # randomly shuffled,
            # punctuation at the end is preserved
        elif sent.endswith('?'):  # question mark
            sent = re.sub(r'\?$', ' ?', sent).split()
            copy_of_sent = sent.copy()
            while copy_of_sent == sent:
                random.shuffle(copy_of_sent)
            all_sentences_tokenized_scrambled.append([tok for tok in copy_of_sent])
            copy_of_sent.remove('?')
            all_sentences_tokenized_scrambled_punctuation_final.append(copy_of_sent + ['?'])
        elif sent.endswith('!'):
            sent = re.sub(r'!$', ' !', sent).split()
            copy_of_sent = sent.copy()
            while copy_of_sent == sent:
                random.shuffle(copy_of_sent)
            all_sentences_tokenized_scrambled.append([tok for tok in copy_of_sent])
            copy_of_sent.remove('!')
            all_sentences_tokenized_scrambled_punctuation_final.append(copy_of_sent + ['!'])

        # remove all punctuation and save the shuffled sentences
        all_sentences_tokenized_scrambled_no_punc.append(re.sub(r'[.,!?;:]', '', ' '.join(copy_of_sent)))

    # save all scrambled sentences in a df
    df_sents['scrambled'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled]
    df_sents['scrambled_final_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_punctuation_final]
    df_sents['scrambled_no_punc'] = all_sentences_tokenized_scrambled_no_punc

    # write the df to file
    df_sents.to_csv('Unpermuted Datasets/readability_data_shufled.tsv', sep='\t', header=True, encoding='utf-8', index_label='index')
    print('Scrambled data saved to file.')


if __name__ == '__main__':
    path_to_test_file = 'Unpermuted Datasets/test.tsv'
    shuffle_sentences_randomly(path_to_test_file)
    path_to_train_file = 'Unpermuted Datasets/train.tsv'
    shuffle_sentences_randomly(path_to_train_file)
    path_to_dev_file = 'Unpermuted Datasets/dev.tsv'
    shuffle_sentences_randomly(path_to_dev_file)
