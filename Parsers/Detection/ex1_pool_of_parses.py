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
    df_dev_c = pd.read_csv('Data/Trees/train_C.tsv', encoding='utf-8', sep='\t', header=0)

    pool = {t for t in df_dev_c['tree']}

    df_test_c = pd.read_csv('Data/Trees/test_C.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_ar = pd.read_csv('Data/Trees/test_AR.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_vr = pd.read_csv('Data/Trees/test_VR.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_vt = pd.read_csv('Data/Trees/test_VT.tsv', encoding='utf-8', sep='\t', header=0)
    label_df_vt = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                              'test_shuffled_random_all_and_verbs_and_tendencies.tsv', encoding='utf-8', sep='\t',
                              header=0)

    # get the gold labels for each individual portion of the datasets
    c_labels_gold = ['correct' for _ in range(len(df_test_c))]
    ar_labels_gold = ['incorrect' for _ in range(len(df_test_ar))]
    vr_labels_gold = ['incorrect' for _ in range(len(df_test_vr))]
    vt_labels_gold = ['correct' if label == 'correct' else 'incorrect' for label in label_df_vt['general_error_label']]

    # get the predicted labels for each individual portion of the datasets (for calculating accuracy)
    predicted_labels_c = ['correct' if tree in pool else 'incorrect' for tree in df_test_c['tree']]
    predicted_labels_ar = ['correct' if tree in pool else 'incorrect' for tree in df_test_ar['tree']]
    predicted_labels_vr = ['correct' if tree in pool else 'incorrect' for tree in df_test_vr['tree']]
    predicted_labels_vt = ['correct' if tree in pool else 'incorrect' for tree in df_test_vt['tree']]

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
    print('VT_C\n___')
    get_metrics(vt_labels_gold, predicted_labels_vt)
    print()


if __name__ == '__main__':
    experiment_1_basic_classification_from_pool()
