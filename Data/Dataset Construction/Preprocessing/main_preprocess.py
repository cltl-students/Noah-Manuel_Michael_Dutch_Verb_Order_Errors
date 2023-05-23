# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 23.05.2023
# Main script for preprocessing the base datasets for the synthetic datasets

from preprocess_readability_corpus import preprocess_readability_corpus
from preprocess_lassy_corpus import preprocess_lassy_corpus
from preprocess_wainot_corpus import preprocess_wainot_corpus

if __name__ == '__main__':
    preprocess_readability_corpus()
    preprocess_lassy_corpus()
    preprocess_wainot_corpus()
