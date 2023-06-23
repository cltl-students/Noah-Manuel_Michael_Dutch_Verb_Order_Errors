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


def get_results_on_VT_exp_6():
    """

    :return:
    """
    df_test = pd.read_csv('../../Data/Dataset_Construction/'
                          'Permuted_Datasets/test_shuffled_random_all_and_verbs_and_tendencies.tsv',
                          sep='\t', encoding='utf-8', header=0)

    test_gold = ['correct' if label == 'correct' else 'incorrect' for label in df_test['general_error_label']]

    test_VT_predicted_bertje_AR = read_predictions('Predictions/predictions_bertje_AR_no_punc_VT.txt')
    test_VT_predicted_bertje_VR = read_predictions('Predictions/predictions_bertje_VR_no_punc_VT.txt')
    test_VT_predicted_robbert_AR = read_predictions('Predictions/predictions_robbert_AR_no_punc_VT.txt')
    test_VT_predicted_robbert_VR = read_predictions('Predictions/predictions_robbert_VR_no_punc_VT.txt')

    print('BERTje AR on VT:')
    get_metrics(test_gold, test_VT_predicted_bertje_AR)

    print('BERTje VR on VT:')
    get_metrics(test_gold, test_VT_predicted_bertje_VR)

    print('RobBERT AR on VT:')
    get_metrics(test_gold, test_VT_predicted_robbert_AR)

    print('RobBERT VR on VT:')
    get_metrics(test_gold, test_VT_predicted_robbert_VR)


if __name__ == '__main__':
    print('___ AR & VR ___')
    get_results_experiment_6()
    print('___ VT ___')
    get_results_on_VT_exp_6()
