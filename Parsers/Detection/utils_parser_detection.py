# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 09.06.2023
# utils for parser experiments (detection)

import json
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score, fbeta_score


def read_in_json_data_and_convert_to_str_sequence(path, split, dataset, spacy=False):
    """
    Read in the tuple information from the json files. If the tuples were created with spacy, access the list of tuples
    for each sentence via a key. Otherwise, access every line and obtain the list of tuples. Convert each tuple, which
    is in list form, into a str sequence of the form 0_PoS. Join all tuples of a single sentence together into a str
    sequence for further processing with CountVectorizer.
    :param str path: path to directory
    :param str split: train, dev, test
    :param str dataset: AR, VR, C
    :param bool spacy: False for access to tuples without key, True for access to tuples with key
    :return: list json_data: a list of all tuple sequences for all sentences
    """
    json_data = []

    with open(f'{path}_{split}_{dataset}.json') as infile:
        content = infile.readlines()
        for line in content:
            sent_tuples = json.loads(line.strip())
            if spacy:  # if processed with spacy on SURF the tuples are saved in a dictionary, so we have to access the
                # values through the key
                for key in ['no_punc', 'scrambled_no_punc', 'verbs_random_no_punc', 'tendencies_no_punc']:
                    try:  # retrieve the list of lists, join the tuples together to be of the form 0_PoS, then join
                        # everything to be one str sequence
                        json_data.append(' '.join([(str(tup[0]) + '_' + str(tup[1])) for tup in sent_tuples[key]]))
                    except KeyError:
                        continue
            else:  # otherwise, each line is already the list of tuples
                json_data.append(' '.join([(str(tup[0]) + '_' + str(tup[1])) for tup in sent_tuples]))

    print(f'{split.title()} {dataset.title()} tuples read in.')

    return json_data


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
