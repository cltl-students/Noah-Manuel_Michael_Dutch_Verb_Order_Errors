# Noah-Manuel Michael
# Created: 20.05.2023
# Last updated: 31.05.2023
# Permute only verb positions in data for dataset creation

import re
import random
import pandas as pd
import spacy


def shuffle_only_verb_positions():
    """
    Read in sentences and get a single permutation of only the verb tokens' positions.
    :return: None
    """
    for split in ['test', 'dev', 'train']:
        df = pd.read_csv(f'Permuted_Datasets/{split}_shuffled_random.tsv', sep='\t', header=0, encoding='utf-8')

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
                if token in ['VERB', 'AUX']:
                    masked_sentence.append(token.text)
                else:
                    masked_sentence.append(0)  # get the sentence template with only the verbs

            copy_of_masked_sentence = masked_sentence.copy()

            if [el for el in masked_sentence if el != 0]:  # (include sentence only if parser was able to detect a verb)
                while copy_of_masked_sentence == masked_sentence:  # make sure the shuffle is different from the original
                    random.shuffle(copy_of_masked_sentence)  # shuffle only the verbs

                # define a cache from which to fill in the previously masked words in their original order
                cache = [token for i, token in enumerate(full_sentence) if token != masked_sentence[i]]
                for i, element in enumerate(copy_of_masked_sentence):
                    if element == 0 and len(cache) > 1:
                        copy_of_masked_sentence[i] = cache.pop(0)  # fill the template with the rest of the tokens in their
                        # correct order
                copy_of_masked_sentence.remove(0)
                if sent.endswith('.'):
                    copy_of_masked_sentence = copy_of_masked_sentence + ['.']
                elif sent.endswith('?'):
                    copy_of_masked_sentence = copy_of_masked_sentence + ['?']
                elif sent.endswith('!'):
                    copy_of_masked_sentence = copy_of_masked_sentence + ['!']

                final_sent = ' '.join(copy_of_masked_sentence)
                cleaned_sent = re.sub(r'_', ' ', final_sent)  # get rid of _ that frog tokenizer adds for set expressions

                # remove unnecessary spaces introduced by frog tokenizer
                opening_chars = ['(', '“', '‘']
                closing_chars = [')', ':', ';', '’', '”']
                for char in opening_chars:
                    cleaned_sent = re.sub(r'{} '.format(re.escape(char)), char, cleaned_sent)
                for char in closing_chars:
                    cleaned_sent = re.sub(r' {}'.format(re.escape(char)), char, cleaned_sent)

                all_scrambled_sents.append(cleaned_sent)
                all_scrambled_sents_no_punc.append(re.sub(r'[.,!?;:]', '', cleaned_sent))
                df_filter.append(True)

            else:
                all_scrambled_sents.append('')
                all_scrambled_sents_no_punc.append('')
                df_filter.append(False)

        df['verbs_scrambled_final_punc'] = all_scrambled_sents
        df['verbs_scrambled_no_punc'] = all_scrambled_sents_no_punc
        df_new = df.loc[df_filter]  # filter df for only those sentences where the tokenizer was able to find a verb

        df_new.to_csv(f'{split}', sep='\t', header=True, index=False, encoding='utf-8')


if __name__ == '__main__':
    shuffle_only_verb_positions()
