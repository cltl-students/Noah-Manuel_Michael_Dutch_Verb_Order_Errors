# Noah-Manuel Michael
# Created: 06.05.2023
# Last updated: 12.05.2023
# Get results for experiment 6 (fine-tuning for detection)

import pandas as pd
from utils_transformer_detection import read_predictions, get_metrics


def get_results_experiment_6():
    """

    :return:
    """
    df_test = pd.read_csv('../../Data/Dataset_Construction/Permuted_Datasets/test_shuffled_random_all_and_verbs.tsv',
                          sep='\t', encoding='utf-8', header=0)

    test_gold = ['correct' for _ in range(len(df_test['original']))] + \
                ['incorrect' for _ in range(len(df_test['original']))]

    test_AR_predicted_bertje = read_predictions('Predictions/predictions_bertje_no_punc_AR.txt')
    test_VR_predicted_bertje = read_predictions('Predictions/predictions_bertje_no_punc_VR.txt')
    test_AR_predicted_robbert = read_predictions('Predictions/predictions_robbert_no_punc_AR.txt')
    test_VR_predicted_robbert = read_predictions('Predictions/predictions_robbert_no_punc_VR.txt')
    test_AR_predicted_gpt2 = read_predictions('Predictions/predictions_gpt2_no_punc_AR.txt')
    test_VR_predicted_gpt2 = read_predictions('Predictions/predictions_gpt2_no_punc_VR.txt')

    print('BERTje on AR:')
    get_metrics(test_gold, test_AR_predicted_bertje)

    print('BERTje on VR:')
    get_metrics(test_gold, test_VR_predicted_bertje)

    print('RobBERT on AR:')
    get_metrics(test_gold, test_AR_predicted_robbert)

    print('RobBERT on VR:')
    get_metrics(test_gold, test_VR_predicted_robbert)

    print('GPT-2 on AR:')
    get_metrics(test_gold, test_AR_predicted_gpt2)

    print('GPT-2 on VR:')
    get_metrics(test_gold, test_VR_predicted_gpt2)


if __name__ == '__main__':
    get_results_experiment_6()
