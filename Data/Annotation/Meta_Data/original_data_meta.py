# Noah-Manuel Michael
# Created: 26.05.2023
# Last updated: 26.05.2023
# Get the maximum len of learner sentences to limit the maximum len of sentences in the synthetic datasets

import pandas as pd
import spacy
from collections import Counter


def get_metadata_from_original_dataset():
    df = pd.read_csv('../Data/leerder_corpus_KU_preprocessed.tsv', sep='\t', encoding='utf-8', header=0)

    print('Levels in the data:')
    print(Counter(df['Level']))

    texts = {text for text in df['Content']}

    print('Number of different texts in the data:')
    print(len(texts))

    print('Languages in the original data:')
    print(len(set(df['Language'])))
    for language in sorted(set(df['Language'])):
        print(language)

    nlp = spacy.load('nl_core_news_lg')

    lengths_of_sentences = []

    for text in texts:
        doc = nlp(text)
        for sent in doc.sents:
            lengths_of_sentences.append(len(sent))

    distribution = sorted(Counter(lengths_of_sentences).items())

    with open('sentence_length_distribution_in_original.tsv', 'w') as outfile:
        outfile.write('SentenceLength\tSentenceCount\n')
        for length, amount in distribution:
            outfile.write(f'{length}\t{amount}\n')
    print('Distribution of sentence length saved in file for further processing.')


if __name__ == '__main__':
    get_metadata_from_original_dataset()
