# Noah-Manuel Michael
# Created: 10.04.2023
# Last updated: 12.05.2023
# Extract the sentences to annotate

import pandas as pd
import os


def extract_sentences_from_leerder_data():
    """

    :return:
    """
    df = pd.read_csv('Annotation/Unpermuted Datasets/leerder_corpus_KU_preprocessed.tsv', sep='\t', encoding='utf-8', header=0,
                     keep_default_na=False)

    if 'leerder_erroneous_sentences.tsv' not in os.listdir('Annotation/Unpermuted Datasets'):
        with open('Annotation/Unpermuted Datasets/leerder_erroneous_sentences.tsv', 'w') as outfile:
            outfile.write('Index\tLevel\tLanguage\tContent\tSentence\tErrors\tErroneous_words\tComment\n')

    with open('Annotation/Unpermuted Datasets/leerder_erroneous_sentences.tsv', encoding='utf-8') as infile:
        content = infile.readlines()
        if len(content) > 1:
            for line in content[1:]:
                last_line = line.strip().split('\t')
                last_index = int(last_line[0])
        else:
            last_index = 0

    for i, row in df.iloc[last_index+1:].iterrows():
        index = row['Index']
        level = row['Level']
        language = row['Language']
        content = row['Content']
        errors = row['Errors']
        erroneous_words = row['Erroneous_words']
        comment = row['Comment']

        print('Content:')
        print(content)
        print()
        print('Errors:')
        print(errors)
        print()
        print('Erroneous words:')
        print(erroneous_words)
        print()
        print('Comment:')
        print(comment)
        print()

        original_sentence = input('Sentence to annotate: ')
        print()

        content = content.replace('\n', ' ')
        errors_no_new_lines = errors.replace('\n', ' ')
        comment_no_new_lines = comment.replace('\n', ' ')

        if original_sentence != 'skip':
            with open('Annotation/Unpermuted Datasets/leerder_erroneous_sentences.tsv', 'a', encoding='utf-8') as outfile:
                outfile.write(f'{index}\t{level}\t{language}\t{content}\t{original_sentence}\t'
                              f'{errors_no_new_lines}\t{erroneous_words}\t{comment_no_new_lines}\n')


if __name__ == '__main__':
    extract_sentences_from_leerder_data()
