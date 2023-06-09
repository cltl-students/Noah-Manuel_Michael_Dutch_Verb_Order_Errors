# Noah-Manuel Michael
# Created: 20.05.2023
# Last updated: 09.06.2023
# Permute only verb positions in data for dataset creation
# Executed on SURF Research Cloud

import re
import random
import pandas as pd
import spacy


def shuffle_only_verb_positions(split):
    """
    Read in sentences and get a single permutation of only the verb tokens' positions.
    :return: None
    """
    df = pd.read_csv(f'Permuted_Datasets/{split}_shuffled_random_all.tsv', sep='\t', header=0, encoding='utf-8')

    nlp = spacy.load('nl_core_news_lg')

    all_scrambled_sents = []
    all_scrambled_sents_no_punc = []
    df_filter = []

    for sent in df['spaced']:
        doc = nlp(sent)

        full_sentence = []
        masked_sentence = []

        for token in doc:
            full_sentence.append(token.text)  # get the full sentence
            if token.pos_ in ['VERB', 'AUX']:
                masked_sentence.append(token.text)
            else:
                masked_sentence.append(0)  # get the sentence template with only the verbs

        masked_sentence = masked_sentence[:-1]  # ignore the final punctuation token for shuffling

        # make a copy to shuffle verbs, original is needed to check whether the shuffled != original and to get
        # the cache
        copy_of_masked_sentence = masked_sentence.copy()

        if [el for el in masked_sentence if el != 0] and len(masked_sentence) > 1:  # include sentence only if parser
            # was able to detect a verb and the sentence consists of more than 1 verb + 1 punc
            while copy_of_masked_sentence == masked_sentence:  # make sure the shuffle is different from the
                # original
                random.shuffle(copy_of_masked_sentence)  # shuffle only the verbs, this includes that punctuation
                # cannot be the only change which would result in the original sentences being equal to shuffled

            # define a cache from which to fill in the previously masked words in their original order
            # get all tokens that are 0 in the unrandomized masked sentence
            cache = [token for i, token in enumerate(full_sentence[:-1]) if token != masked_sentence[i]]
            for i, element in enumerate(copy_of_masked_sentence):
                if element == 0:  # and len(cache) > 1:  # up until last token in the cache which is the punc token
                    copy_of_masked_sentence[i] = cache.pop(0)  # fill the template with the rest of the tokens in
                    # their correct order
            # copy_of_masked_sentence.remove(0)  # remove the last remaining 0 from the restored copy

            # check in which punc token the original strings of sentences ended
            if sent.endswith('.'):  # add punctuation as last token (that way it does never occur in any position
                # but [-1])
                copy_of_masked_sentence = copy_of_masked_sentence + ['.']
            elif sent.endswith('?'):
                copy_of_masked_sentence = copy_of_masked_sentence + ['?']
            elif sent.endswith('!'):
                copy_of_masked_sentence = copy_of_masked_sentence + ['!']

            final_sent = ' '.join(copy_of_masked_sentence)

            no_punc_sent = re.sub(r'[^\w ]', '', final_sent)
            no_punc_sent = re.sub(r'  +', ' ', no_punc_sent)

            all_scrambled_sents.append(final_sent)
            all_scrambled_sents_no_punc.append(no_punc_sent)
            df_filter.append(True)

        else:
            all_scrambled_sents.append('')
            all_scrambled_sents_no_punc.append('')
            df_filter.append(False)

    df['verbs_random_punc_final'] = all_scrambled_sents
    df['verbs_random_no_punc'] = all_scrambled_sents_no_punc

    df_new = df.loc[df_filter]  # filter df for only those sentences where the tokenizer was able to find a verb
    df_new = df_new.drop(columns='index')
    df_new = df_new.reset_index(drop=True)

    df_new.to_csv(f'Permuted_Datasets/{split}_shuffled_random_all_and_verbs.tsv', sep='\t', encoding='utf-8',
                  header=True, index_label='index')

    print(f'{split.title()} split verbs shuffled randomly and written to file.')


if __name__ == '__main__':
    for split in ['test', 'dev', 'train']:
        shuffle_only_verb_positions(split)
