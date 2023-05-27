# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Main script for preprocessing the base datasets for the synthetic datasets

import train_dev_test_split
import get_spaced_punc_sents
import get_no_punc_sents
from preprocess_readability_corpus import preprocess_readability_corpus
from preprocess_lassy_corpus import preprocess_lassy_corpus
from preprocess_wainot_corpus import preprocess_wainot_corpus
from preprocess_leipzig_corpora import preprocess_leipzig_corpora


if __name__ == '__main__':
    print('Preprocessing Edia readability corpus.')
    preprocess_readability_corpus()
    print('Edia readability corpus preprocessed.')

    print('Preprocessing Lassy small corpus.')
    preprocess_lassy_corpus()
    print('Lassy small corpus preprocessed.')

    print('Preprocessing wainot corpus.')
    preprocess_wainot_corpus()
    print('Wainot corpus preprocessed.')

    print('Preprocessing Leipzig corpora.')
    preprocess_leipzig_corpora()
    print('Leipzig corpora preprocessed.')

    train_dev_test_split.get_datasplit()
    print('Data split into train, dev, and test split.')

    get_spaced_punc_sents.get_spaced_sentences()
    print('Spaces introduced according to spaCy tokenizer.')
    get_no_punc_sents.get_no_punc_sentences()
    print('Version of sentences without punctuation saved.')
