# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 24.06.2023
# Get all results of all models on all datasets (parsers, ML approach)

import pickle
import pandas as pd
from utils_parser_detection import read_in_json_data_and_convert_to_str_sequence, get_metrics


def check_all_classifiers_on_all_datasets(train):
    """
    Check all classifiers on all datasets.
    :param str train:
    :return:
    """
    df_VT = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                        'test_shuffled_random_all_and_verbs_and_tendencies.tsv',
                        encoding='utf-8', header=0, sep='\t')

    with open(f'Classifiers/vectorizer_{train}.pkl', 'rb') as infile:
        vectorizer = pickle.load(infile)

    with open(f'Classifiers/logreg_{train}.pkl', 'rb') as infile:
        logreg = pickle.load(infile)

    if not any(['simple' in train, 'spacy' in train]):
        test_C = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'C')
        if not 'VR' in train:
            test_AR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'AR')
        test_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'VR')
        test_VT = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'VT')
    elif 'simple' in train:
        test_C = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                               'test', 'C')
        if not 'VR' in train:
            test_AR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                    'test', 'AR')
        test_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                'test', 'VR')
        test_VT = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                'test', 'VT')
    else:
        test_C = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'C',
                                                               spacy=True)
        if not 'VR' in train:
            test_AR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'AR',
                                                                    spacy=True)
        test_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'VR',
                                                                spacy=True)
        test_VT = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'VT',
                                                                spacy=True)

    if not 'VR' in train:
        test_AR_full = test_C + test_AR
        test_AR_full_vectorized = vectorizer.transform(test_AR_full)
        predictions_AR = logreg.predict(test_AR_full_vectorized)
        with open(f'Data/Tuples/Predictions/predictions_{train}_train_AR_test.txt', 'w') as outfile:
            for pred in predictions_AR:
                outfile.write(f'{pred}\n')
        gold_labels_AR = ['correct' for _ in range(len(test_C))] + ['incorrect' for _ in range(len(test_AR))]
        print(f'Results for {train} trained classifier on AR dataset:')
        get_metrics(gold_labels_AR, predictions_AR)
        print('\n______\n')

    test_VR_full = test_C + test_VR
    test_VR_full_vectorized = vectorizer.transform(test_VR_full)
    predictions_VR = logreg.predict(test_VR_full_vectorized)
    with open(f'Data/Tuples/Predictions/predictions_{train}_train_VR_test.txt', 'w') as outfile:
        for pred in predictions_VR:
            outfile.write(f'{pred}\n')
    gold_labels_VR = ['correct' for _ in range(len(test_C))] + \
                     ['incorrect' for _ in range(len(test_VR))]
    print(f'Results for {train} trained classifier on VR dataset:')
    get_metrics(gold_labels_VR, predictions_VR)
    print('\n______\n')

    test_VT_vectorized = vectorizer.transform(test_VT)
    predictions_VT = logreg.predict(test_VT_vectorized)
    with open(f'Data/Tuples/Predictions/predictions_{train}_train_VT_test.txt', 'w') as outfile:
        for pred in predictions_VT:
            outfile.write(f'{pred}\n')
    gold_labels_VT = ['correct' if label == 'correct' else 'incorrect' for label in df_VT['general_error_label']]
    print(f'Results for {train} trained classifier on VT dataset:')
    get_metrics(gold_labels_VT, predictions_VT)
    print('\n______\n')


if __name__ == '__main__':
    for train in ['disco_AR', 'disco_VR', 'disco_simple_AR', 'disco_simple_VR', 'spacy_AR', 'spacy_VR']:
        check_all_classifiers_on_all_datasets(train)
