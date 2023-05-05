# Noah-Manuel Michael
# 30.04.2023
# Annotation software

import pandas as pd
import os

df = pd.read_csv('Annotation/Data/erroneous_sentences.tsv', sep='\t', encoding='utf-8', header=0, keep_default_na=False)

if 'annotated_data.tsv' not in os.listdir('Annotation/Data'):
    with open('Annotation/Data/annotated_data.tsv', 'w') as outfile:
        outfile.write('Index\tLevel\tLanguage\tSentence\tNormalized\tCorrected\tClause_type\tVerb_type\tError_type\t'
                      'Clause_structure\tConfidence\n')

with open('Annotation/Data/annotated_data.tsv', encoding='utf-8') as infile:
    content = infile.readlines()
    if len(content) > 1:
        for line in content[1:]:
            last_line = line.strip().split('\t')
            last_index = int(last_line[0])
    else:
        last_index = 0

print('You will now start/resume the annotation process.\n'
      'Reminder:\n'
      '"Correct Sentence" asks for the fully corrected version of the sentence.\n'
      '"Error Sentence" asks you to reintroduce the verb order error the learner made to the corrected sentence.\n')

for i, row in df.iterrows():
    index = row['Index']
    if index > last_index:
        level = row['Level']
        language = row['Language']
        content = row['Content']
        errors = row['Errors']
        erroneous_words = row['Erroneous_words']
        comment = row['Comment']
        original_sentence = row['Sentence']

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
        print(f'Please correct this sentence:\n'
              f'{original_sentence}')
        print()

        corrected_sentence = input('Correct sentence: ')
        normalized_sentence = input('Error sentence:   ')
        print('\n'
              '0: Completely unsure\n'
              '1: Very unsure\n'
              '2: Slightly unsure\n'
              '3: Slightly sure\n'
              '4: Very sure\n'
              '5: Completely sure'
              '\n')
        confidence = input('Confidence: ')
        num_erroneous_clauses = int(input('How many clauses exhibit errors in this sentence? '))
        for n in range(num_erroneous_clauses):
            print(f'You are now annotating error number {n+1}.')
            print('\n'
                  'm (main clause)\n'
                  's (subclause)\n'
                  'p (polar question)\n'
                  'o (open question)'
                  '\n')
            clause_type = input('Clause type: ')
            print('\n'
                  'f (finite verb)\n'
                  'nf (non-finite verb)\n'
                  'ic (infinitival complement)\n'
                  'pc (prepositional complement)\n'
                  '0 (word order error is not concerned with verb order)'
                  '\n')
            verb_type = input('Verb type: ')

            if verb_type == '0':
                error_type = 'misc'
            elif verb_type == 'ic' or verb_type == 'pc':
                error_type = 'post-verbal'
            elif verb_type == 'nf' or clause_type == 's':
                error_type = 'verb-final'
            elif any([clause_type == 'm', clause_type == 'o']):
                error_type = 'verb-second'
            elif clause_type == 'p':
                error_type = 'verb-first'

            print()
            print(f'Error type: {error_type}')

            if verb_type == '0':
                clause_structure = '0'
            else:
                print()
                print(normalized_sentence)
                clause_structure = input('Clause structure: ')

            with open('Annotation/Data/annotated_data.tsv', 'a', encoding='utf-8') as outfile:
                outfile.write(f'{index}\t{level}\t{language}\t{original_sentence}\t{normalized_sentence}\t'
                              f'{corrected_sentence}\t{clause_type}\t{verb_type}\t{error_type}\t{clause_structure}\t'
                              f'{confidence}\n')

        print()
