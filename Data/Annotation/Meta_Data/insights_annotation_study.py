# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 19.06.2023
# Extract meta data from leerder data

import pandas as pd
from collections import Counter


def extract_insights_from_annotated_data():
    """

    :return:
    """
    df = pd.read_csv('../Data/final_annotated_data.tsv', sep='\t', encoding='utf-8', header=0)

    print('Languages:')
    print(Counter(df['Language']))

    print('Levels:')
    print(Counter(df['Level']))

    print('Error types:')
    print(Counter(df['Error_type']))

    print('Verb types:')
    print(Counter(df['Verb_type']))

    print('Finite verb errors:')
    print(Counter([row['Error_type'] for i, row in df.iterrows() if row['Verb_type'] == 'f']))

    print('Target hypotheses challenged by both native speakers:')
    for i, row in df.iterrows():
        if type(row['A2']) != float and type(row['A1']) != float:
            print(row['Annotated'])
            print('Old:')
            print(row['Corrected'])
            print('New:')
            print(row['Target_hypothesis'])
            print('_________________________')

    print('Finite verb main clause error clause structures:')
    for clause_structure in [row['Clause_structure'] for i, row in df.iterrows() if row['Verb_type'] == 'f' and
                                                                                    row['Clause_type'] == 'm']:
        print(clause_structure)

    print('_________________________')

    print('Finite verb subordinate clause error clause structures:')
    for clause_structure in [row['Clause_structure'] for i, row in df.iterrows() if row['Verb_type'] == 'f' and
                                                                                    row['Clause_type'] == 's']:
        print(clause_structure)


if __name__ == '__main__':
    extract_insights_from_annotated_data()
