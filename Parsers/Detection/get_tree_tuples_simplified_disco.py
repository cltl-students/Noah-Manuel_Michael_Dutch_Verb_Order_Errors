# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 11.06.2023
# Extract tuples from parses as features for classifiers

import re
import pandas as pd
import json


def get_tuples_disco_simple():
    for split in ['test', 'train']:
        for dataset in ['C', 'AR', 'VR']:
            df = pd.read_csv(f'Data/Trees/{split}_{dataset}.tsv', encoding='utf-8', header=0, sep='\t')

            for tree in df['simple_tree']:
                token_tuples = []
                tree = re.sub(r'[(ROOT)\(\)]', '', tree)

                for i, token in enumerate(tree.split()):
                    token_tuples.append((i, token))

                with open(f'Data/Tuples/Disco_simple/tuple_data_simple_disco_{split}_{dataset}.json', 'a') as outfile:
                    json.dump(token_tuples, outfile)
                    outfile.write('\n')


if __name__ == '__main__':
    get_tuples_disco_simple()
