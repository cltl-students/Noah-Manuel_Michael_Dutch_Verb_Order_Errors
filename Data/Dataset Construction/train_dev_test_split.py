# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 24.05.2023
# Unite the preprocessed datasets and split into train, dev, and test datasets

import pandas as pd

# TO-DO: Parse everything with frog to make sure there is at least one verb in the sentence

def get_datasplit():
    """

    :return:
    """
    df_edia = pd.read_csv('Unpermuted Datasets/readability_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_lassy = pd.read_csv('Unpermuted Datasets/lassy_data.tsv', sep='\t', encoding='utf-8', header=0)
    df_wainot = pd.read_csv('Unpermuted Datasets/wainot_data.tsv', sep='\t', encoding='utf-8', header=0)

    # reduce the test data to the levels we see in the actual learner data
    levels_for_test_set = ['A2', 'A2+', 'B1', 'B1+', 'B2', 'B2+', 'C1']
    df_levels_filtered = df_edia.loc[[True if level in levels_for_test_set else False for level in df_edia['level']]]

    df_test = pd.DataFrame({'original': df_levels_filtered['original'].tolist(),  # make new df to get correct indices
                            'level': df_levels_filtered['level'].tolist()})

    leftover_sents = []  # use the sents that are outside the level range for train and dev data
    for i, row in df_edia.iterrows():
        if row['level'] not in levels_for_test_set:
            leftover_sents.append(row['original'])

    # combine all other data
    combined_data = df_lassy['original'].tolist() + df_wainot['original'].tolist() + leftover_sents

    # get every 10th sentence for dev data
    df_dev = pd.DataFrame([sent for i, sent in enumerate(combined_data) if i % 10 == 0], columns=['original'])

    # all other sents for train data
    df_train = pd.DataFrame([sent for i, sent in enumerate(combined_data) if i % 10 != 0], columns=['original'])

    df_test.to_csv('Unpermuted Datasets/test.tsv', sep='\t', index_label='index', encoding='utf-8')
    df_dev.to_csv('Unpermuted Datasets/dev.tsv', sep='\t', index_label='index', encoding='utf-8')
    df_train.to_csv('Unpermuted Datasets/train.tsv', sep='\t', index_label='index', encoding='utf-8')


if __name__ == '__main__':
    get_datasplit()
