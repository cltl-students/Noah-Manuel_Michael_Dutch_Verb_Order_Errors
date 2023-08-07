# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 11.06.2023
# Check if the particular tree structure is present in the trees accepted as correct

import pandas as pd
from utils_parser_detection import get_metrics


def experiment_1_basic_classification_from_pool():
    """

    :return:
    """
    df_dev_c = pd.read_csv('Data/Trees/train_Correct.tsv', encoding='utf-8', sep='\t', header=0)

    pool = {t for t in df_dev_c['tree']}

    df_test_Correct = pd.read_csv('Data/Trees/test_Correct.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_Rand = pd.read_csv('Data/Trees/test_Rand.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_Verbs = pd.read_csv('Data/Trees/test_Verbs.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_Info = pd.read_csv('Data/Trees/test_Info.tsv', encoding='utf-8', sep='\t', header=0)
    label_df_Info = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                              'test_shuffled_random_all_and_verbs_and_tendencies.tsv', encoding='utf-8', sep='\t',
                              header=0)
    df_test_Learn = pd.read_csv('Data/Trees/test_Learn.tsv', encoding='utf-8', sep='\t', header=0)

    # get the gold labels for each individual portion of the datasets
    Correct_labels_gold = ['correct' for _ in range(len(df_test_Correct))]
    Rand_labels_gold = ['incorrect' for _ in range(len(df_test_Rand))]
    Verbs_labels_gold = ['incorrect' for _ in range(len(df_test_Verbs))]
    Info_labels_gold = ['correct' if label == 'correct' else 'incorrect' for label in label_df_Info['general_error_label']]
    Learn_labels_gold = ['incorrect' for _ in range(len(df_test_Learn))]

    # get the predicted labels for each individual portion of the datasets (for calculating accuracy)
    predicted_labels_Correct = ['correct' if tree in pool else 'incorrect' for tree in df_test_Correct['tree']]
    predicted_labels_Rand = ['correct' if tree in pool else 'incorrect' for tree in df_test_Rand['tree']]
    predicted_labels_Verbs = ['correct' if tree in pool else 'incorrect' for tree in df_test_Verbs['tree']]
    predicted_labels_Info = ['correct' if tree in pool else 'incorrect' for tree in df_test_Info['tree']]
    predicted_labels_Learn = ['correct' if tree in pool else 'incorrect' for tree in df_test_Learn['tree']]

    # get the labels for the combined datasets (shuffled portion + correct portion for precision, recall, f)
    gold_labels_Rand_Correct = Rand_labels_gold + Correct_labels_gold
    predicted_labels_Rand_Correct = predicted_labels_Rand + predicted_labels_Correct

    gold_labels_Verbs_Correct = Verbs_labels_gold + Correct_labels_gold
    predicted_labels_Verbs_Correct = predicted_labels_Verbs + predicted_labels_Correct

    print('Rand_C\n___')
    get_metrics(gold_labels_Rand_Correct, predicted_labels_Rand_Correct)
    print()
    print('Verbs_C\n___')
    get_metrics(gold_labels_Verbs_Correct, predicted_labels_Verbs_Correct)
    print()
    print('Info_C\n___')
    get_metrics(Info_labels_gold, predicted_labels_Info)
    print()
    print('Learn\n___')
    get_metrics(Learn_labels_gold, predicted_labels_Learn)
    print()


if __name__ == '__main__':
    experiment_1_basic_classification_from_pool()
