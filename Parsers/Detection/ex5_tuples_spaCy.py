# Noah-Manuel Michael
# Created: 08.06.2023
# Last updated: 08.06.2023
# Train classifiers based on tuple information (spacy)

import pickle
from utils_parser_experiments import read_in_json_data_and_convert_to_str_sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def run_experiment_5(datasplit):
    """

    :param str datasplit:
    :return:
    """
    train_spacy_C = read_in_json_data_and_convert_to_str_sequence('Tuples/Spacy/tuple_data_spacy', 'train', 'C',
                                                                  spacy=True)
    train_spacy_split = read_in_json_data_and_convert_to_str_sequence('Tuples/Spacy/tuple_data_spacy', 'train',
                                                                      datasplit, spacy=True)

    test_spacy_C = read_in_json_data_and_convert_to_str_sequence('Tuples/Spacy/tuple_data_spacy', 'test', 'C',
                                                                 spacy=True)
    test_spacy_split = read_in_json_data_and_convert_to_str_sequence('Tuples/Spacy/tuple_data_spacy', 'test', datasplit,
                                                                     spacy=True)

    train_spacy_full = train_spacy_C + train_spacy_split
    test_spacy_full = test_spacy_C + test_spacy_split

    vectorizer = CountVectorizer()
    train_spacy_vectorized = vectorizer.fit_transform(train_spacy_full)
    print('Vocabulary built and training data vectorized.')
    test_spacy_vectorized = vectorizer.transform(test_spacy_full)
    print('Test data vectorized.')

    logreg = LogisticRegression(max_iter=1000)
    print('Classifier instantiated.')

    gold_train_spacy_full = ['correct' for _ in range(len(train_spacy_C))] + \
                            ['incorrect' for _ in range(len(train_spacy_split))]

    logreg.fit(train_spacy_vectorized, gold_train_spacy_full)
    print('Classifier fitted.')

    predictions = logreg.predict(test_spacy_vectorized)
    print('Test instances predicted.')

    with open(f'Tuples/Predictions/predictions_spacy_{datasplit}.txt', 'w') as outfile:
        for pred in predictions:
            outfile.write(f'{pred}\n')

    print('Predictions saved to file.')

    with open(f'Tuples/vectorizer_spacy_{datasplit}.pkl', 'wb') as outfile:
        pickle.dump(vectorizer, outfile)
    with open(f'Tuples/logreg_spacy_{datasplit}.pkl', 'wb') as outfile:
        pickle.dump(logreg, outfile)

    print('Classifier and vectorizer saved to file.')


if __name__ == '__main__':
    run_experiment_5('AR')
    run_experiment_5('VR')
