# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 09.06.2023
# utils for transformer experiments (detection)

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def read_predictions(path):
    """
    Read predictions from a prediction file.
    :param path:
    :return:
    """
    with open(path) as infile:
        list_of_predictions = []
        content = infile.readlines()
        for prediction in content:
            list_of_predictions.append(prediction.strip())

    return list_of_predictions


def get_metrics(y_true, y_pred):
    """
    Print the results of the experiments for each test dataset.
    :param y_true: true labels for a dataset
    :param y_pred: predicated labels for a dataset
    :return: None
    """
    matrix = confusion_matrix(y_true, y_pred, labels=['incorrect', 'correct'])

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred,
                                                                     labels=['incorrect', 'correct'])
    _, _, f05, _ = precision_recall_fscore_support(y_true, y_pred, beta=0.5,
                                                                     labels=['incorrect', 'correct'])

    metrics_summary = pd.DataFrame(
        {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'F0.5 Score': f05,
            'Support': support
        },
        index=['incorrect', 'correct'])

    accuracy = accuracy_score(y_true, y_pred)

    macro_micro_average_f05 = (f05[0] + f05[1]) / 2

    print('Confusion matrix [\'incorrect\', \'correct\']:')
    print(matrix)
    print(metrics_summary)
    print('Accuracy:')
    print(accuracy)
    print('Average F0.5:')
    print(macro_micro_average_f05)
