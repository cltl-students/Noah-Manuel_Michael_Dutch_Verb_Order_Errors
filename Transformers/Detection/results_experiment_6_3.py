# Noah-Manuel Michael
# Created: 06.05.2023
# Last updated: 23.06.2023
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

    test_AR_predicted_bertje = read_predictions('Predictions3/predictions_bertje_AR_on_AR3.txt')
    test_AR_VR_predicted_bertje = read_predictions('Predictions3/predictions_bertje_AR_on_VR3.txt')
    test_VR_predicted_bertje = read_predictions('Predictions3/predictions_bertje_VR_on_VR3.txt')
    test_AR_predicted_robbert = read_predictions('Predictions3/predictions_robbert_AR_on_AR3.txt')
    test_AR_VR_predicted_robbert = read_predictions('Predictions3/predictions_robbert_AR_on_VR3.txt')
    test_VR_predicted_robbert = read_predictions('Predictions3/predictions_robbert_VR_on_VR3.txt')
    test_AR_predicted_gpt2 = read_predictions('Predictions3/predictions_gpt2_AR_on_AR3.txt')
    test_AR_VR_predicted_gpt2 = read_predictions('Predictions3/predictions_gpt2_AR_on_VR3.txt')
    test_VR_predicted_gpt2 = read_predictions('Predictions3/predictions_gpt2_VR_on_VR3.txt')

    print('BERTje AR on AR:')
    get_metrics(test_gold, test_AR_predicted_bertje)

    print('BERTje AR on VR:')
    get_metrics(test_gold, test_AR_VR_predicted_bertje)

    print('BERTje VR on VR:')
    get_metrics(test_gold, test_VR_predicted_bertje)

    print('RobBERT AR on AR:')
    get_metrics(test_gold, test_AR_predicted_robbert)

    # for i, gold in enumerate(test_gold):
    #     if gold != test_AR_predicted_robbert[i]:
    #         try:
    #             print(i, gold, test_AR_predicted_robbert[i], df_test['original'][i], df_test['scrambled_no_punc'][i])
    #         except KeyError:
    #             print(i-13586, gold, test_AR_predicted_robbert[i-13586], df_test['original'][i-13586], df_test['scrambled_no_punc'][i-13586])

    print('RobBERT AR on VR:')
    get_metrics(test_gold, test_AR_VR_predicted_robbert)

    print('RobBERT VR on VR:')
    get_metrics(test_gold, test_VR_predicted_robbert)

    print('GPT-2 AR on AR:')
    get_metrics(test_gold, test_AR_predicted_gpt2)

    print('GPT-2 AR on VR:')
    get_metrics(test_gold, test_AR_VR_predicted_gpt2)

    print('GPT-2 VR on VR:')
    get_metrics(test_gold, test_VR_predicted_gpt2)


def get_results_on_VT_exp_6():
    """

    :return:
    """
    df_test = pd.read_csv('../../Data/Dataset_Construction/'
                          'Permuted_Datasets/test_shuffled_random_all_and_verbs_and_tendencies.tsv',
                          sep='\t', encoding='utf-8', header=0)

    test_gold = ['correct' if label == 'correct' else 'incorrect' for label in df_test['general_error_label']]

    test_VT_predicted_bertje_AR = read_predictions('Predictions3/predictions_bertje_AR_on_VT3.txt')
    test_VT_predicted_bertje_VR = read_predictions('Predictions3/predictions_bertje_VR_on_VT3.txt')
    test_VT_predicted_robbert_AR = read_predictions('Predictions3/predictions_robbert_AR_on_VT3.txt')
    test_VT_predicted_robbert_VR = read_predictions('Predictions3/predictions_robbert_VR_on_VT3.txt')
    test_VT_predicted_gpt2_AR = read_predictions('Predictions3/predictions_gpt2_AR_on_VT.txt')
    test_VT_predicted_gpt2_VR = read_predictions('Predictions3/predictions_gpt2_VR_on_VT.txt')

    print('BERTje AR on VT:')
    get_metrics(test_gold, test_VT_predicted_bertje_AR)

    print('BERTje VR on VT:')
    get_metrics(test_gold, test_VT_predicted_bertje_VR)

    print('RobBERT AR on VT:')
    get_metrics(test_gold, test_VT_predicted_robbert_AR)

    print('RobBERT VR on VT:')
    get_metrics(test_gold, test_VT_predicted_robbert_VR)

    print('GPT-2 AR on VT:')
    get_metrics(test_gold, test_VT_predicted_gpt2_AR)

    print('GPT-2 VR on VT:')
    get_metrics(test_gold, test_VT_predicted_gpt2_VR)


if __name__ == '__main__':
    print('___ AR & VR ___')
    get_results_experiment_6()
    print('___ VT ___')
    get_results_on_VT_exp_6()
