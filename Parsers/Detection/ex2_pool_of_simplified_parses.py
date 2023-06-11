# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 11.06.2023
# Check if the particular tree structure is present in the trees accepted as correct

import pandas as pd
from utils_parser_experiments import get_metrics


def experiment_2_basic_classification_from_simplified_pool():
    """

    :return:
    """
    df_dev_c = pd.read_csv('Data/Trees/train_C.tsv', encoding='utf-8', sep='\t', header=0)

    pool = {t for t in df_dev_c['simple_tree']}

    df_test_c = pd.read_csv('Data/Trees/test_C.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_ar = pd.read_csv('Data/Trees/test_AR.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_vr = pd.read_csv('Data/Trees/test_VR.tsv', encoding='utf-8', sep='\t', header=0)

    # get the gold labels for each individual portion of the datasets
    c_labels_gold = ['correct' for _ in range(len(df_test_c))]
    ar_labels_gold = ['incorrect' for _ in range(len(df_test_ar))]
    vr_labels_gold = ['incorrect' for _ in range(len(df_test_vr))]

    # get the predicted labels for each individual portion of the datasets (for calculating accuracy)
    predicted_labels_c = ['correct' if tree in pool else 'incorrect' for tree in df_test_c['simple_tree']]
    predicted_labels_ar = ['correct' if tree in pool else 'incorrect' for tree in df_test_ar['simple_tree']]
    predicted_labels_vr = ['correct' if tree in pool else 'incorrect' for tree in df_test_vr['simple_tree']]

    # get the labels for the combined datasets (shuffled portion + correct portion for precision, recall, f)
    gold_labels_ar_c = ar_labels_gold + c_labels_gold
    predicted_labels_ar_c = predicted_labels_ar + predicted_labels_c

    gold_labels_vr_c = vr_labels_gold + c_labels_gold
    predicted_labels_vr_c = predicted_labels_vr + predicted_labels_c

    print('AR_C\n___')
    get_metrics(gold_labels_ar_c, predicted_labels_ar_c)
    print()
    print('VR_C\n___')
    get_metrics(gold_labels_vr_c, predicted_labels_vr_c)
    print()


if __name__ == '__main__':
    experiment_2_basic_classification_from_simplified_pool()
