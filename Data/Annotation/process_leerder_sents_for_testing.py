# Noah-Manuel Michael
# Created: 07.08.2023
# Last updated: 07.08.2023
# preprocess original learner sentences to test performance of classification approaches

import pandas as pd
import re


def remove_punc_from_learner_sents():
    """

    :return:
    """
    df = pd.read_csv('Data/final_annotated_data.tsv', encoding='utf-8', header=0, sep='\t')

    no_punc_sents = set()

    for sent in df['Normalized']:
        sent = re.sub(r'[^\w ]', '', sent)
        sent = re.sub(r'  +', ' ', sent)
        no_punc_sents.add(sent)

    with open('Data/leerder_sents_no_punc_for_testing.tsv', 'w') as outfile:
        for sent in no_punc_sents:
            outfile.write(f'{sent}\n')


if __name__ == '__main__':
    remove_punc_from_learner_sents()
