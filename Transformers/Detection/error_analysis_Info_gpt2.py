# Noah-Manuel Michael
# Created: 07.08.2023
# Last updated: 07.08.2023
# Look at the misclassified sentences in the Info dataset

import pandas as pd
from utils_transformer_detection import read_predictions
from collections import defaultdict, Counter

predictions = read_predictions('Predictions/predictions_gpt2_AR_on_VT.txt')
predictions3 = read_predictions('Predictions3/predictions_gpt2_AR_on_VT3.txt')

df_Info = pd.read_csv('../../Data/Dataset_Construction/'
                      'Permuted_Datasets/test_shuffled_random_all_and_verbs_and_tendencies.tsv', encoding='utf-8',
                      sep='\t', header=0)

count = defaultdict(list)

df_Info['prediction'] = predictions

for i, row in df_Info.iterrows():
    count[row['detailed_error_label']].extend([row['prediction']])

for errortype, instances in count.items():
    print(errortype, Counter(instances))

