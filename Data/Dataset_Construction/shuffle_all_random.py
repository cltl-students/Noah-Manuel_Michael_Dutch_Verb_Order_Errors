# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 21.05.2023
# Scramble the readability data randomly

import pandas as pd
import re
import random


def shuffle_sentences_randomly():
    """
    Read in the preprocessed data split. Get a single random permutation for each sentence. Save different
    versions of the permuted sentence in a new file.
    Permuted sentence types: scrambled - scrambled with final punctuation - scrambled with no punctuation
    :return: None
    """
    # read in the data split
    for split in ['test', 'dev', 'train']:
        df = pd.read_csv(f'Unpermuted_Datasets/{split}.tsv', sep='\t', header=0, encoding='utf-8')

        # define lists to store the scrambled sentences in
        all_sentences_scrambled = []
        all_sentences_scrambled_punctuation_final = []
        all_sentences_scrambled_no_punc = []

        for sent in df['spaced']:
            # check for punctuation marks at the end of sentences, split the sentences, shuffle, append shuffled
            # sentences to lists
            split_sent = sent.split()  # the spaces were introduced by spaCy tokenizer, so we reconstruct the spaCy toks
            copy_to_be_shuffled = split_sent.copy()

            # make sure the shuffled version != the original version + punctuation is not the only change in the
            # shuffled version (because in the no_punc versions the original and the shuffled would be equal if only
            # punctuation got shuffled
            while re.sub(r'[^\w ]', '', ' '.join(split_sent)) == re.sub(r'[^\w ]', '', ' '.join(copy_to_be_shuffled)):
                random.shuffle(copy_to_be_shuffled)

            all_sentences_scrambled.append(' '.join(copy_to_be_shuffled))  # randomly shuffled
            # remove all punctuation and save the shuffled sentences
            sent_no_punc = re.sub(r'[^\w ]', '', ' '.join(copy_to_be_shuffled))
            sent_no_punc = re.sub(r'^ +', '', sent_no_punc)
            all_sentences_scrambled_no_punc.append(re.sub(r'  +', ' ', sent_no_punc))

            if sent.endswith('.'):  # full stop
                try:  # some sentences seem to have been mistokenized by spacy (i.e., there is no space before the final
                    # punctuation token, therefore we mark those here with a 0
                    copy_to_be_shuffled.remove('.')
                    # randomly shuffled, punctuation at the end is preserved
                    all_sentences_scrambled_punctuation_final.append(' '.join(copy_to_be_shuffled) + ' .')
                except ValueError:
                    all_sentences_scrambled_punctuation_final.append(0)
            elif sent.endswith('?'):  # question mark
                try:
                    copy_to_be_shuffled.remove('?')
                    all_sentences_scrambled_punctuation_final.append(' '.join(copy_to_be_shuffled) + ' ?')
                except ValueError:
                    all_sentences_scrambled_punctuation_final.append(0)
            elif sent.endswith('!'):  # exclamation mark
                try:
                    copy_to_be_shuffled.remove('!')
                    all_sentences_scrambled_punctuation_final.append(' '.join(copy_to_be_shuffled) + ' !')
                except ValueError:
                    all_sentences_scrambled_punctuation_final.append(0)

        # save all scrambled sentences in a df
        df['scrambled'] = all_sentences_scrambled
        df['scrambled_final_punc'] = all_sentences_scrambled_punctuation_final
        df['scrambled_no_punc'] = all_sentences_scrambled_no_punc
        df = df.drop(columns='index')

        # filter out the mistokenized sentences
        df_filtered = df.loc[[False if value == 0 else True for value in df['scrambled_final_punc']]]
        df_filtered.reset_index(drop=True, inplace=True)

        # write the df to file
        df_filtered.to_csv(f'Permuted_Datasets/{split}_shuffled_random_all.tsv', sep='\t', header=True,
                           encoding='utf-8', index_label='index')
        print('Scrambled data saved to file.')


if __name__ == '__main__':
    shuffle_sentences_randomly()
