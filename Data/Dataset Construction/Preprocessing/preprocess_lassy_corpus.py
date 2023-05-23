# Noah-Manuel Michael
# Created: 21.05.2023
# Last updated: 23.05.2023
# Get Lasssy small sentences for dataset construction

import glob
import re
import pandas as pd


def preprocess_lassy_corpus():
    lassy_sents = []

    file_list = glob.glob('../Unpermuted Datasets/lassy_corpus/*.sents')

    for file in file_list:
        with open(f'{file}', encoding='utf-8') as infile:
            for line in infile.readlines():
                lassy_sents.append(re.sub(r'.*\|', '', line.strip()))

    lassy_sents_no_spaces = []

    begin_chars = ['(', '\'', '"', '‘', '“']
    end_chars = [';', ':', '.', '?', '!', ',', ')', '\'', '"', '’', '”', '%']

    for sent in lassy_sents:
        if len(sent) > 10 and re.match(r'^[A-Z].*[.!?]$', sent):
            for char in begin_chars:
                sent = re.sub(r'{} '.format(re.escape(char)), char, sent)
            for char in end_chars:
                sent = re.sub(r' {}'.format(re.escape(char)), char, sent)
            sent = re.sub(r'([;:?!,\'"’”‘“])([;:?!,\'"’”‘“])', r'\1 \2', sent)
            lassy_sents_no_spaces.append(sent)

    df = pd.DataFrame(lassy_sents_no_spaces, columns=['original'])

    df.to_csv('../Unpermuted Datasets/lassy_data.tsv', sep='\t', index_label='index', encoding='utf-8')


if __name__ == '__main__':
    preprocess_lassy_corpus()
