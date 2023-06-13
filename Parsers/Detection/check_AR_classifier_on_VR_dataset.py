# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 07.06.2023
# Explore the performance of AR classifiers on VR datasets to determine whether training on AR is already enough

import pickle
from utils_parser_detection import read_in_json_data_and_convert_to_str_sequence, get_metrics


def check_AR_train_VR_predict():
    """
    Check the performance of a classifer trained on AR data on the VR test dataset. Use the unsimplified disco-dop
    tuples (as the performance with the simplified tuples was almost the same and this only serves illustrational
    purposes).
    :return:
    """
    with open('Classifiers/vectorizer_disco_AR.pkl', 'rb') as infile:
        vectorizer_AR = pickle.load(infile)

    with open('Classifiers/logreg_disco_AR.pkl', 'rb') as infile:
        logreg_AR = pickle.load(infile)

    test_disco_correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test',
                                                                       'C')
    test_disco_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'VR')

    test_disco_VR_full = test_disco_correct + test_disco_VR

    test_disco_VR_full_vectorized = vectorizer_AR.transform(test_disco_VR_full)

    predictions_AR_classifier_on_VR_data = logreg_AR.predict(test_disco_VR_full_vectorized)

    with open('Data/Tuples/Predictions/predictions_disco_AR_train_VR_test.txt', 'w') as outfile:
        for pred in predictions_AR_classifier_on_VR_data:
            outfile.write(f'{pred}\n')

    gold_labels_VR = ['correct' for _ in range(len(test_disco_correct))] + \
                     ['incorrect' for _ in range(len(test_disco_VR))]

    print('Results for AR trained classifier on VR dataset (untouched tree tuples):')
    get_metrics(gold_labels_VR, predictions_AR_classifier_on_VR_data)


if __name__ == '__main__':
    check_AR_train_VR_predict()
