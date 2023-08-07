# Noah-Manuel Michael
# Created: 17.06.2023
# Last updated: 18.06.2023
# Analyze human evaluations of annotation data

import pandas as pd


def change_analysis_based_on_native_speaker_verification():
    """

    :return:
    """
    df_a1 = pd.read_csv('Annotations_A1/Annotations/leerder_annotated_data_A1.tsv', sep='\t', header=0,
                        encoding='utf-8')
    df_a2 = pd.read_csv('Annotations_A2/Annotations/leerder_annotated_data_A2.tsv', sep='\t', header=0,
                        encoding='utf-8')

    df_a1['A2'] = df_a2['A2']
    df_combined = df_a1

    final_sents = []

    for i, row in df_combined.iterrows():
        if type(row['A1']) != float and type(row['A2']) != float:
            print('Sentence:')
            print(row['Sentence'])
            print('Target hypothesis:')
            print(row['Corrected'])
            print('Reintroduced verb error:')
            print(row['Normalized'])
            print('A1:')
            print(row['A1'])
            print('A2:')
            print(row['A2'])
            final_sent = input('Accepted Correction:\n')
            final_sents.append(final_sent)
            df_combined.at[i, 'Normalized'] = input('Normalized sentence:\n')
            print(row['Clause_structure'])
            analysis_change_needed = input('Do you want to change the analysis of the sentence [y/n]?\n')
            if analysis_change_needed == 'y':
                df_combined.at[i, 'Clause_type'] = input('Clause_type:\n')
                df_combined.at[i, 'Verb_type'] = input('Verb_type:\n')
                df_combined.at[i, 'Error_type'] = input('Error_type:\n')
                df_combined.at[i, 'Clause_structure'] = input('Clause_structure:\n')
            print()
        else:
            final_sents.append(row['Corrected'])

    df_combined['Target_hypothesis'] = final_sents

    df_combined.to_csv('Data/final_annotated_data.tsv', sep='\t', encoding='utf-8', header=True)


if __name__ == '__main__':
    change_analysis_based_on_native_speaker_verification()
