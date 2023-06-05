# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 05.06.2023
# Unite the parsed data (executed several notebook in parallel in SURF)

import pandas as pd


def unite_parsed_tree_data():
    """

    :return:
    """
    for verbs in ['', '_verbs']:
        df_1 = pd.read_csv(f'Data/dev_incorrect{verbs}_parsed1.tsv', sep='\t', encoding='utf-8', header=0)
        df_2 = pd.read_csv(f'Data/dev_incorrect{verbs}_parsed2.tsv', sep='\t', encoding='utf-8', header=0)
        df_3 = pd.read_csv(f'Data/dev_incorrect{verbs}_parsed3.tsv', sep='\t', encoding='utf-8', header=0)
        df_4 = pd.read_csv(f'Data/dev_incorrect{verbs}_parsed4.tsv', sep='\t', encoding='utf-8', header=0)

        frames = [df_1, df_2, df_3, df_4]

        contatenated_df = pd.concat(frames).reset_index(drop=True)
        contatenated_df = contatenated_df.drop(columns='index')

        if verbs == '':
            contatenated_df.to_csv('Data/dev_AR.tsv', index_label='index', encoding='utf-8', sep='\t')
        else:
            contatenated_df.to_csv('Data/dev_VR.tsv', index_label='index', encoding='utf-8', sep='\t')


if __name__ == '__main__':
    unite_parsed_tree_data()
