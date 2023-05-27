# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 25.05.2023
# Format sentences to have no punctuation

import re
import pandas as pd


def get_no_punc_sentences():
    """

    :return:
    """
    for split in ['test', 'dev', 'train']:
        df = pd.read_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', header=0)

        no_punc_sents = []

        for sent in df['spaced']:
            sent = re.sub(r'[^\w ]', '', sent)
            sent = re.sub(r'  +', ' ', sent)
            no_punc_sents.append(sent)

        df['no_punc'] = no_punc_sents

        df.to_csv(f'../Unpermuted_Datasets/{split}.tsv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    get_no_punc_sentences()
