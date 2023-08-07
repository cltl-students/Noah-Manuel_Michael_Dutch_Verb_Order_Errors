# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 11.06.2023
# Unite the parsed data (executed several notebook in parallel in SURF)

import pandas as pd


def unite_parsed_tree_data():
    """
    Unite the parsed dev files from SURF. Up to 20,000 sentences were processed in parallel (4x) resulting in 4
    different files that need to be concatenated for the whole pool.
    :return:
    """
    for path in ['Data/Trees/Trees/train_correct_parsed', 'Data/Trees/Trees/train_incorrect_parsed',
                 'Data/Trees/Trees/train_incorrect_verbs_parsed']:
        df_1 = pd.read_csv(path + '1.tsv', sep='\t', encoding='utf-8', header=0)
        df_2 = pd.read_csv(path + '2.tsv', sep='\t', encoding='utf-8', header=0)
        df_3 = pd.read_csv(path + '3.tsv', sep='\t', encoding='utf-8', header=0)
        df_4 = pd.read_csv(path + '4.tsv', sep='\t', encoding='utf-8', header=0)

        frames = [df_1, df_2, df_3, df_4]

        contatenated_df = pd.concat(frames).reset_index(drop=True)
        contatenated_df = contatenated_df.drop(columns='index')

        if path == 'Data/Trees/Trees/train_correct_parsed':
            contatenated_df.to_csv('Data/train_Correct.tsv', index_label='index', encoding='utf-8', sep='\t')
        elif path == 'Data/Trees/Trees/train_incorrect_parsed':
            contatenated_df.to_csv('Data/train_Rand.tsv', index_label='index', encoding='utf-8', sep='\t')
        else:
            contatenated_df.to_csv('Data/train_Verbs.tsv', index_label='index', encoding='utf-8', sep='\t')


if __name__ == '__main__':
    unite_parsed_tree_data()
