# Noah-Manuel Michael
# Created: 12.06.2023
# Last updated: 12.06.2023
# Let target hypotheses be confirmed by native speaker

import pandas as pd
import os


def confirm_target_hypotheses():
    df = pd.read_csv('leerder_annotated_data_clean.tsv', sep='\t', encoding='utf-8', header=0)

    annotator = input('Your name: ')
    print()

    if 'save.txt' not in os.listdir():
        start_idx = 0
    else:
        with open('save.txt') as infile:
            start_idx = int(infile.readline().strip())

    if f'leerder_annotated_data_{annotator}.tsv' not in os.listdir():
        with open(f'leerder_annotated_data_{annotator}.tsv', 'w') as outfile:
            outfile.write(f'Index\tLevel\tLanguage\tSentence\tNormalized\tCorrected\tClause_type\tVerb_type\t'
                          f'Error_type\tClause_structure\tAnnotated\t{annotator}\n')

    print('You will now see a pair of sentences. The first sentence is a sentence that was produced by a learner of '
          'Dutch as a second language. Your task is to evaluate whether the second sentence is a possible correction of'
          ' the first sentence. For many sentences, there is more than one possible correction. Your task is only to '
          'evaluate whether the corrected version presented to you is one of them. For your evaluation, '
          'keep in mind that ideally, the corrected version of the sentence should stick as closely as possible to the '
          'original wording used by the learner. If you agree with the proposed target hypothesis (the corrected '
          'sentence), you will proceed to the next sentence pair. If you do not agree, please provide a corrected '
          'version of the learner sentence yourself. Sometimes the corrected sentence does not cover the whole '
          'original sentence produced by the learner. This is usually the case if the learner sentence consists of a '
          'combination of phrases or sentences that would typically be separated. In this case, please only evaluate '
          'the part that is covered by the corrected sentence.\n')

    for i, row in df[start_idx:].iterrows():
        print('Original sentence:')
        print(row['Sentence'])
        print()
        print('Corrected sentence (target hypothesis):')
        print(row['Corrected'])
        print()

        accept = input('Second sentence is an acceptable target hypothesis [0/1]?\n')
        if int(accept) == 0:
            new_hypothesis = input('Your corrected version:\n')
        else:
            new_hypothesis = 'NA'
        print()

        with open('save.txt', 'w') as outfile:
            outfile.write(str(i+1))

        with open(f'leerder_annotated_data_{annotator}.tsv', 'a') as outfile:
            for column in row:
                outfile.write(f'{column}\t')
            outfile.write(f'{new_hypothesis}\n')


if __name__ == '__main__':
    confirm_target_hypotheses()
