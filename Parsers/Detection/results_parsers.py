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
    df_Info = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/'
                        'test_shuffled_random_all_and_verbs_and_tendencies.tsv',
                        encoding='utf-8', header=0, sep='\t')

    with open(f'Classifiers/vectorizer_{train}.pkl', 'rb') as infile:
        vectorizer = pickle.load(infile)

    with open(f'Classifiers/logreg_{train}.pkl', 'rb') as infile:
        logreg = pickle.load(infile)

    if not any(['simple' in train, 'spacy' in train]):
        test_Correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'Correct')
        if not 'Verbs' in train:
            test_Rand = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'Rand')
        test_Verbs = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'Verbs')
        test_Info = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'Info')
        test_Learn = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'Learn')
    elif 'simple' in train:
        test_Correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                               'test', 'Correct')
        if not 'Verbs' in train:
            test_Rand = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                    'test', 'Rand')
        test_Verbs = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                'test', 'Verbs')
        test_Info = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                'test', 'Info')
        test_Learn = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                  'test', 'Learn')
    else:
        test_Correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'Correct',
                                                               spacy=True)
        if not 'Verbs' in train:
            test_Rand = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'Rand',
                                                                    spacy=True)
        test_Verbs = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'Verbs',
                                                                spacy=True)
        test_Info = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'Info',
                                                                spacy=True)
        test_Learn = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/tuple_data_spacy', 'test', 'Learn',
                                                                  spacy=True)

    if not 'Verbs' in train:
        test_Rand_full = test_Correct + test_Rand
        test_Rand_full_vectorized = vectorizer.transform(test_Rand_full)
        predictions_Rand = logreg.predict(test_Rand_full_vectorized)
        with open(f'Data/Tuples/Predictions/predictions_{train}_train_Rand_test.txt', 'w') as outfile:
            for pred in predictions_Rand:
                outfile.write(f'{pred}\n')
        gold_labels_Rand = ['correct' for _ in range(len(test_Correct))] + ['incorrect' for _ in range(len(test_Rand))]
        print(f'Results for {train} trained classifier on Rand dataset:')
        get_metrics(gold_labels_Rand, predictions_Rand)
        print('\n______\n')

    test_Verbs_full = test_Correct + test_Verbs
    test_Verbs_full_vectorized = vectorizer.transform(test_Verbs_full)
    predictions_Verbs = logreg.predict(test_Verbs_full_vectorized)
    with open(f'Data/Tuples/Predictions/predictions_{train}_train_Verbs_test.txt', 'w') as outfile:
        for pred in predictions_Verbs:
            outfile.write(f'{pred}\n')
    gold_labels_Verbs = ['correct' for _ in range(len(test_Correct))] + \
                     ['incorrect' for _ in range(len(test_Verbs))]
    print(f'Results for {train} trained classifier on Verbs dataset:')
    get_metrics(gold_labels_Verbs, predictions_Verbs)
    print('\n______\n')

    test_Info_vectorized = vectorizer.transform(test_Info)
    predictions_Info = logreg.predict(test_Info_vectorized)
    with open(f'Data/Tuples/Predictions/predictions_{train}_train_Info_test.txt', 'w') as outfile:
        for pred in predictions_Info:
            outfile.write(f'{pred}\n')
    gold_labels_Info = ['correct' if label == 'correct' else 'incorrect' for label in df_Info['general_error_label']]
    print(f'Results for {train} trained classifier on Info dataset:')
    get_metrics(gold_labels_Info, predictions_Info)
    print('\n______\n')

    test_Learn_vectorized = vectorizer.transform(test_Learn)
    predictions_Learn = logreg.predict(test_Learn_vectorized)
    with open(f'Data/Tuples/Predictions/predictions_{train}_train_Learn_test.txt', 'w') as outfile:
        for pred in predictions_Info:
            outfile.write(f'{pred}\n')
    gold_labels_Learn = ['incorrect' for _ in range(len(test_Learn))]
    print(f'Results for {train} trained classifier on Learn dataset:')
    get_metrics(gold_labels_Learn, predictions_Learn)
    print('\n______\n')


if __name__ == '__main__':
    for train in ['disco_Rand', 'disco_Verbs', 'disco_simple_Rand', 'disco_simple_Verbs', 'spacy_Rand', 'spacy_Verbs']:
        check_all_classifiers_on_all_datasets(train)
