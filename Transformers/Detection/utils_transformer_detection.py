# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 09.06.2023
# utils for transformer experiments (detection)

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score, fbeta_score


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

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    average_f1 = (f1[0] + f1[1]) / 2
    macro_f05 = fbeta_score(y_true, y_pred, beta=0.5, average='macro')
    micro_f05 = fbeta_score(y_true, y_pred, beta=0.5, average='micro')
    average_f05 = (f05[0] + f05[1]) / 2

    print(f'Confusion matrix [\'incorrect\', \'correct\']:\n{matrix}')
    print(metrics_summary)
    print(f'Macro F1:\n{macro_f1}')
    print(f'Micro F1:\n{micro_f1}')
    print(f'Average F1:\n{average_f1}')
    print(f'Macro F0.5:\n{macro_f05}')
    print(f'Micro F0.5:\n{micro_f05}')
    print(f'Average F0.5:\n{average_f05}')
    print(f'Accuracy:\n{accuracy}')

    print('_____________________')


def get_VT_metrics(y_true, y_pred):
    """
    Print the results of the experiments for each test dataset.
    :param y_true: true labels for a dataset
    :param y_pred: predicated labels for a dataset
    :return: None
    """
    label_list = ['maincorrect', 'mainrandom', 'maininitial', 'mainmedialbeforesubj', 'mainmedialaftersubj',
                  'mainfinalbeforenonfinite', 'mainfinal', 'subcorrect', 'subrandom', 'subinitial',
                  'submedialbeforesubj', 'submedialaftersubj']

    matrix = confusion_matrix(y_true, y_pred, labels=label_list)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=label_list)
    _, _, f05, _ = precision_recall_fscore_support(y_true, y_pred, beta=0.5, labels=label_list)

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

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    average_f1 = sum([score for score in f1]) / 12
    macro_f05 = fbeta_score(y_true, y_pred, beta=0.5, average='macro')
    micro_f05 = fbeta_score(y_true, y_pred, beta=0.5, average='micro')
    average_f05 = sum([score for score in f05]) / 12

    print(f'Confusion matrix:\n{label_list}\n{matrix}')
    print(metrics_summary)
    print(f'Macro F1:\n{macro_f1}')
    print(f'Micro F1:\n{micro_f1}')
    print(f'Average F1:\n{average_f1}')
    print(f'Macro F0.5:\n{macro_f05}')
    print(f'Micro F0.5:\n{micro_f05}')
    print(f'Average F0.5:\n{average_f05}')
    print(f'Accuracy:\n{accuracy}')

    print('_____________________')
