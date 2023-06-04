# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 03.06.2023
# Check if the particular tree structure is present in the trees accepted as correct

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def experiment_2_basic_classification_from_simplified_pool():
    """

    :return:
    """
    df_dev_correct = pd.read_csv('Data/dev_correct_simplified.tsv', encoding='utf-8', sep='\t', header=0)

    all_correct_trees_dev = {t for t in df_dev_correct['simple_tree']}

    df_test_correct = pd.read_csv('Data/test_correct_simplified.tsv', encoding='utf-8', sep='\t', header=0)
    df_test_incorrect = pd.read_csv('Data/test_incorrect_simplified.tsv', encoding='utf-8', sep='\t', header=0)

    all_correct_trees_test = [t for t in df_test_correct['simple_tree']]
    all_incorrect_trees_test = [t for t in df_test_incorrect['simple_tree']]

    original_labels = ['correct' for _ in range(len(all_correct_trees_test))] + \
                      ['incorrect' for _ in range(len(all_incorrect_trees_test))]

    predicted_labels = ['correct' if sent in all_correct_trees_dev else 'incorrect'
                        for sent in (all_correct_trees_test + all_incorrect_trees_test)]

    matrix = confusion_matrix(original_labels, predicted_labels)
    scores_f1 = precision_recall_fscore_support(original_labels, predicted_labels, labels=['incorrect', 'correct'])
    scores_f05 = precision_recall_fscore_support(original_labels, predicted_labels, beta=0.5,
                                                 labels=['incorrect', 'correct'])
    accuracy = accuracy_score(original_labels, predicted_labels)

    print(matrix)
    print(scores_f1)
    print(scores_f05)
    print(accuracy)


if __name__ == '__main__':
    experiment_2_basic_classification_from_simplified_pool()
