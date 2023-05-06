import pandas as pd
import re
import spacy
import random

# read in the original texts from the readability corpus
df_texts = pd.read_csv('dataset_cefr_dutch.csv', sep=',', header=0, encoding='utf-8')

# get rid of new lines and double spaces, also convert to set so every text is present only once (in original data every
# text appears three times
all_texts = set(re.sub(r'\n', ' ', text) for text in df_texts['text_plain'])
all_texts = [re.sub(r'  ', ' ', text) for text in all_texts]

nlp = spacy.load('nl_core_news_sm')

# retrieve a list of all sentences that are longer than 10 characters (otherwise mostly single words, not interesting
# for word order
list_of_all_sentences = []
for i, text in enumerate(all_texts):
    doc = nlp(text)
    for sent in doc.sents:
        if len(sent.text) > 10:
            list_of_all_sentences.append(sent.text)
print('All texts split into sentences.')

# save the original sentences in a dataframe before starting to shuffle them
df_orig = pd.DataFrame(list_of_all_sentences, columns=['Original'])

# tokenize sentence, scramble randomly
sentences_for_scramble = df_orig['Original'].tolist()

# define lists to store the scrambled sentences in
all_sentences_tokenized_scrambled = []
all_sentences_tokenized_scrambled_punctuation_final = []
all_sentences_tokenized_scrambled_lower = []
all_sentences_tokenized_scrambled_punctuation_final_lower = []
all_sentences_tokenized_scrambled_no_punc = []
all_sentences_tokenized_scrambled_no_punc_lower = []

for sent in sentences_for_scramble:
    # add empty spaces before and after commas, so they are recognized as tokens of their own
    if ', ' in sent:
        sent = re.sub(r', ', ' , ', sent)

    # check for punctuation marks at the end of sentences, tokenize the sentences, append shuffled sentences to list

    # dot
    if sent.endswith('.'):
        sent = re.sub(r'\.', ' .', sent).split()
        random.shuffle(sent)
        all_sentences_tokenized_scrambled.append([tok for tok in sent])  # randomly shuffled
        all_sentences_tokenized_scrambled_lower.append([tok.lower() for tok in sent])  # randomly shuffled, lowercase
        sent.remove('.')
        all_sentences_tokenized_scrambled_punctuation_final.append(sent + ['.'])  # randomly shuffled, punctuation at
        # the end is preserved
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
df_orig['Scramble'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled]
df_orig['Scramble_lower'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_lower]
df_orig['Scramble_final_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_punctuation_final]
df_orig['Scramble_final_punc_lower'] = [' '.join(sent) for sent in
                                        all_sentences_tokenized_scrambled_punctuation_final_lower]
df_orig['Scramble_no_punc'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_no_punc]
df_orig['Scramble_no_punc_lower'] = [' '.join(sent) for sent in all_sentences_tokenized_scrambled_no_punc_lower]

# write the df to file
df_orig.to_csv('scrambled_data.tsv', sep='\t', header=True, encoding='utf-8', index_label='Index')
print('Scrambled data saved to file.')
