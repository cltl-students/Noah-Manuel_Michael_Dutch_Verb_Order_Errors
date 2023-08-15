# Noah-Manuel Michael
# Created: 07.08.2023
# Last updated: 07.08.2023
# Get results of transformer classifiers on Learn data

from utils_transformer_detection import read_predictions, get_metrics


def get_results_tranformer_classifiers_on_Learn():
    """

    :return:
    """
    test_gold = ['incorrect' for _ in range(184)]

    models = ['bertje_Rand_on_Learn', 'bertje_Verbs_on_Learn', 'robbert_Rand_on_Learn', 'robbert_Verbs_on_Learn',
              'gpt2_Rand_on_Learn', 'gpt2_Verbs_on_Learn']
    iterations = ['', '2', '3']

    for model in models:
        for iteration in iterations:
            predictions = read_predictions(f'Predictions_Learn/predictions_{model}{iteration}.txt')
            print(f'{model}{iteration}')
            get_metrics(test_gold, predictions)
            if model == 'gpt2_Rand_on_Learn' and iteration == '':
                for i, predic in enumerate(predictions):
                    if predic != 'incorrect':
                        print(i)


if __name__ == '__main__':
    get_results_tranformer_classifiers_on_Learn()