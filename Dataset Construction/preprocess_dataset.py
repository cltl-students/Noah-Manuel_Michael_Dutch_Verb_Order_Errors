import pandas as pd
import re
from collections import defaultdict, Counter


def main():
    # read in the original dataset
    df = pd.read_csv('dataset_cefr_dutch.csv', sep=',', header=0, encoding='utf-8')

    # get rid of new line characters and double spaces
    all_texts = [re.sub(r'\n', ' ', text) for text in df['text_plain']]
    all_texts = [re.sub(r'  +', ' ', text) for text in all_texts]
    df['text_normalized'] = all_texts

    # check which texts appear less or more than 3 times
    no_three_annotators = defaultdict(int)
    for key, value in Counter(df['text_normalized']).items():
        if value != 3:
            no_three_annotators[key] = value

    # map levels to points and vice versa
    level_to_points = {'A1': 1, 'A1+': 2, 'A2': 3, 'A2+': 4, 'B1': 5, 'B1+': 6, 'B2': 7, 'B2+': 8, 'C1': 9, 'C1+': 10,
                       'C2': 11, 'C2+': 12}
    points_to_level = {value: key for key, value in level_to_points.items()}

    # get the points for each text
    level_points = []
    for level in df['cefr_level']:
        level_points.append(level_to_points[level])

    # add the points to the dataframe
    df['level_points'] = level_points

    # get a mapping of texts and sum of points
    texts_to_sum = defaultdict(int)
    for i, row in df.iterrows():
        texts_to_sum[row['text_normalized']] += row['level_points']

    # calculate the average for each text and map back to level
    texts_to_average_level = defaultdict(str)
    for text, summ in texts_to_sum.items():
        if text not in no_three_annotators.keys():
            texts_to_average_level[text] = points_to_level[round(summ / 3)]
        else:  # get the actual amount of annotators if they are not 3
            texts_to_average_level[text] = points_to_level[round(summ / no_three_annotators[text])]

    df_processed = pd.DataFrame([text for text in texts_to_average_level.keys()], columns=['text'])
    df_processed['level'] = [level for level in texts_to_average_level.values()]

    print(df_processed)

    df_processed.to_csv('dataset_cefr_dutch_preprocessed.tsv', sep='\t', index_label='Index', encoding='utf-8',
                        header=True)


if __name__ == '__main__':
    main()
