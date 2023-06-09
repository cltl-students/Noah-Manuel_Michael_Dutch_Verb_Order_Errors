# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 09.06.2023
# Scramble sentences randomly

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
    for split in ['dev', 'train']:
        df = pd.read_csv(f'Unpermuted_Datasets/{split}.tsv', sep='\t', header=0, encoding='utf-8')

        # define lists to store the scrambled sentences in
        all_sentences_scrambled = []
        all_sentences_scrambled_punctuation_final = []
        all_sentences_scrambled_no_punc = []

        # to filter out sentences that either were mistokenized or only consist of 1 token + 1 punc token
        df_filter = []

        for sent in df['spaced']:
            if sent.endswith(' .') or sent.endswith(' ?') or sent.endswith(' !'):  # some sents mistokenized by spaCy so
                # we check again, then split the sentences, shuffle, append shuffled sentences to lists
                split_sent = sent.split()  # the spaces were introduced by spaCy tokenizer, so we reconstruct the spaCy
                # tokens
                copy_to_be_shuffled = split_sent[:-1].copy()  # take the sentence except the final punc

                # make sure the shuffled version != the original version + punctuation is not the only change in the
                # shuffled version (because in the no_punc versions the original and the shuffled would be equal if only
                # punctuation got shuffled
                if len(copy_to_be_shuffled) > 1:
                    while copy_to_be_shuffled == split_sent[:-1]:
                        random.shuffle(copy_to_be_shuffled)

                    # reinsert the punctuation token at a random position
                    copy_of_copy = copy_to_be_shuffled.copy()
                    random_position_for_removed_punc_token = random.randint(0, len(split_sent))
                    copy_of_copy.insert(random_position_for_removed_punc_token, split_sent[-1])
                    all_sentences_scrambled.append(' '.join(copy_of_copy))  # randomly shuffled

                    # remove all punctuation and save the shuffled sentences
                    sent_no_punc = re.sub(r'[^\w ]', '', ' '.join(copy_of_copy))
                    sent_no_punc = re.sub(r'^ +', '', sent_no_punc)
                    all_sentences_scrambled_no_punc.append(re.sub(r'  +', ' ', sent_no_punc))

                    # reappend the punc token at the end
                    all_sentences_scrambled_punctuation_final.append(' '.join(copy_to_be_shuffled + [split_sent[-1]]))

                    df_filter.append(True)
                else:
                    all_sentences_scrambled.append('')
                    all_sentences_scrambled_no_punc.append('')
                    all_sentences_scrambled_punctuation_final.append('')
                    df_filter.append(False)

            else:
                all_sentences_scrambled.append('')
                all_sentences_scrambled_no_punc.append('')
                all_sentences_scrambled_punctuation_final.append('')
                df_filter.append(False)

        # save all scrambled sentences in a df
        df['scrambled'] = all_sentences_scrambled
        df['scrambled_final_punc'] = all_sentences_scrambled_punctuation_final
        df['scrambled_no_punc'] = all_sentences_scrambled_no_punc
        df_filtered = df.loc[df_filter]
        df_filtered = df_filtered.drop(columns='index')
        df_filtered = df_filtered.reset_index(drop=True)

        # write the df to file
        df_filtered.to_csv(f'Permuted_Datasets/{split}_shuffled_random_all.tsv', sep='\t', header=True,
                           encoding='utf-8', index_label='index')

        print(f'Scrambled {split} data saved to file.')


if __name__ == '__main__':
    shuffle_sentences_randomly()
