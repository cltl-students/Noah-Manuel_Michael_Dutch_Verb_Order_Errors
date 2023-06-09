# Noah-Manuel Michael
# Created: 07.06.2023
# Last updated: 07.06.2023
# Compute the results with the predictionn files

from utils_parser_experiments import get_metrics, read_predictions, read_in_json_data_and_convert_to_str_sequence


def get_results_experiment_3():
    """

    :return:
    """
    test_disco_correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test',
                                                                       'C')
    test_disco_AR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'AR')
    test_disco_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco/tuple_data_disco', 'test', 'VR')

    labels_ar = ['correct' for _ in range(len(test_disco_correct))] + ['incorrect' for _ in range(len(test_disco_AR))]
    labels_vr = ['correct' for _ in range(len(test_disco_correct))] + ['incorrect' for _ in range(len(test_disco_VR))]

    predictions_ar = read_predictions('Data/Tuples/Predictions/predictions_disco_AR.txt')
    predictions_vr = read_predictions('Data/Tuples/Predictions/predictions_disco_VR.txt')

    print('Experiment 3 Results on AR:')
    get_metrics(labels_ar, predictions_ar)

    print('Experiment 3 Results on VR:')
    get_metrics(labels_vr, predictions_vr)


def get_results_experiment_4():
    """

    :return:
    """
    test_disco_simple_correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/'
                                                                              'tuple_data_simple_disco', 'test', 'C')
    test_disco_simple_AR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/'
                                                                         'tuple_data_simple_disco', 'test', 'AR')
    test_disco_simple_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Disco_simple/'
                                                                         'tuple_data_simple_disco', 'test', 'VR')

    labels_ar_simple = ['correct' for _ in range(len(test_disco_simple_correct))] + \
                       ['incorrect' for _ in range(len(test_disco_simple_AR))]
    labels_vr_simple = ['correct' for _ in range(len(test_disco_simple_correct))] + \
                       ['incorrect' for _ in range(len(test_disco_simple_VR))]

    predictions_ar_simple = read_predictions('Data/Tuples/Predictions/predictions_disco_simple_AR.txt')
    predictions_vr_simple = read_predictions('Data/Tuples/Predictions/predictions_disco_simple_VR.txt')

    print('Experiment 4 Results on AR:')
    get_metrics(labels_ar_simple, predictions_ar_simple)

    print('Experiment 4 Results on VR:')
    get_metrics(labels_vr_simple, predictions_vr_simple)


def get_results_experiment_5():
    """

    :return:
    """
    test_spacy_correct = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/'
                                                                              'tuple_data_spacy', 'test', 'C')
    test_spacy_AR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/'
                                                                         'tuple_data_spacy', 'test', 'AR')
    test_spacy_VR = read_in_json_data_and_convert_to_str_sequence('Data/Tuples/Spacy/'
                                                                         'tuple_data_spacy', 'test', 'VR')

    labels_ar_spacy = ['correct' for _ in range(len(test_spacy_correct))] + \
                      ['incorrect' for _ in range(len(test_spacy_AR))]
    labels_vr_spacy = ['correct' for _ in range(len(test_spacy_correct))] + \
                      ['incorrect' for _ in range(len(test_spacy_VR))]

    predictions_ar_spacy = read_predictions('Data/Tuples/Predictions/predictions_spacy_AR.txt')
    predictions_vr_spacy = read_predictions('Data/Tuples/Predictions/predictions_spacy_VR.txt')

    print('Experiment 5 Results on AR:')
    get_metrics(labels_ar_spacy, predictions_ar_spacy)

    print('Experiment 5 Results on VR:')
    get_metrics(labels_vr_spacy, predictions_vr_spacy)


if __name__ == '__main__':
    get_results_experiment_3()
    get_results_experiment_4()
    get_results_experiment_5()
