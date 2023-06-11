# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 11.06.2023
# Train classifiers based on tuple information

import pickle
from utils_parser_experiments import read_in_json_data_and_convert_to_str_sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def run_experiment_4(datasplit):
    """

    :param str datasplit:
    :return:
    """
    train_disco = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                'train', 'C') + \
                  read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                                'train', datasplit)

    test_disco = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                               'test', 'C') + \
                 read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/tuple_data_simple_disco',
                                                               'test', datasplit)

    vectorizer = CountVectorizer()
    train_disco_vectorized = vectorizer.fit_transform(train_disco)
    print('Vocabulary built and training data vectorized.')
    test_disco_vectorized = vectorizer.transform(test_disco)
    print('Test data vectorized.')

    logreg = LogisticRegression(max_iter=1000)
    print('Classifier instantiated.')

    logreg.fit(train_disco_vectorized, (['correct' for _ in range(int(len(train_disco) / 2))] +
                                        ['incorrect' for _ in range(int(len(train_disco) / 2))]))
    print('Classifier fitted.')

    predictions = logreg.predict(test_disco_vectorized)
    print('Test instances predicted.')

    with open(f'Data/Tuples/Predictions/predictions_disco_simple_{datasplit}.txt', 'w') as outfile:
        for pred in predictions:
            outfile.write(f'{pred}\n')

    print('Predictions saved to file.')

    with open(f'Classifiers/vectorizer_disco_simple_{datasplit}.pkl', 'wb') as outfile:
        pickle.dump(vectorizer, outfile)
    with open(f'Classifiers/logreg_disco_simple_{datasplit}.pkl', 'wb') as outfile:
        pickle.dump(logreg, outfile)

    print('Classifier and vectorizer saved to file.')


if __name__ == '__main__':
    run_experiment_4('AR')
    run_experiment_4('VR')
