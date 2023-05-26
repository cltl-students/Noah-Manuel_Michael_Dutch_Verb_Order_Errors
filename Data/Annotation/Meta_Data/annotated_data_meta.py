# Noah-Manuel Michael
# Created: 30.04.2023
# Last updated: 12.05.2023
# Extract meta data from leerder data

import pandas as pd
from collections import Counter


def extract_meta_data_from_leerder_data():
    """

    :return:
    """
    df = pd.read_csv('../Data/leerder_annotated_data.tsv', sep='\t', keep_default_na=False)

    print(Counter(df['Language']))


if __name__ == '__main__':
    extract_meta_data_from_leerder_data()
