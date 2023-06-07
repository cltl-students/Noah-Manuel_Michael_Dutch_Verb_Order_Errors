# Noah-Manuel Michael
# Created: 06.06.2023
# Last updated: 06.06.2023
# Extract tuples from parses as features for classifiers

import re
import pandas as pd
import json


def get_tuples_disco():
    for split in ['test', 'dev']:
        for dataset in ['C', 'AR', 'VR']:
            df = pd.read_csv(f'Data/Trees/{split}_{dataset}.tsv', encoding='utf-8', header=0, sep='\t')

            for tree in df['tree']:
                token_tuples = []
                tree = re.sub(r'[(ROOT)\(\)]', '', tree)  # remove root and brackets
                tree = re.sub(r'[0-9]', ' ', tree)  # remove original leaves (original position information)
                tree = re.sub('let', '', tree)  # remove punctuation tokens
                tree = re.sub(r'  +', ' ', tree)  # remove additional space characters

                for i, token in enumerate(tree.split()):  # split string sequence on spaces and create position tuples
                    token_tuples.append((i, token))

                with open(f'Data/Tuples/Disco/tuple_data_disco_{split}_{dataset}.json', 'a') as outfile:
                    json.dump(token_tuples, outfile)
                    outfile.write('\n')


if __name__ == '__main__':
    get_tuples_disco()
