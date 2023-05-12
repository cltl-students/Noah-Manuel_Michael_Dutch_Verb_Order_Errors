# Noah-Manuel Michael
# 04.04.2023
# Annotation reader for KU Leuven Corpus
# Manual steps before this script was used:
# 1. Replace all present '</tr>' with ''
# 2. Replace all '<tr>' with '</tr><tr>'
# 3. Delete the first '</tr>'
# 4. Add '</tr>' before '</table>'

import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict

with open('Annotation/Data/corpus_NT2_ILT.html', encoding='utf-8') as infile:
    html = infile.read()

soup = BeautifulSoup(html, 'html.parser')

table = soup.find('table', {'class': 'small'})
rows = table.find_all('tr')

data = []

for row in rows:
    columns = row.find_all('td')
    columns = [col.text.strip() for col in columns]
    data.append(columns)

df_dictionary = defaultdict(list)

for row_data in data:
    df_dictionary['Instance'].append(row_data[0])
    df_dictionary['Level'].append(row_data[1])
    df_dictionary['Language'].append(row_data[2])
    df_dictionary['Content'].append(row_data[3])
    df_dictionary['Error_num'].append(row_data[4])
    df_dictionary['Errors'].append(row_data[5])
    try:
        df_dictionary['Erroneous_words'].append(row_data[6])
        df_dictionary['Comment'].append(row_data[7])
    except IndexError:
        df_dictionary['Erroneous_words'].append('')
        df_dictionary['Comment'].append('')

df = pd.DataFrame.from_dict(df_dictionary)

df.to_csv('Annotation/Data/corpus_data.tsv', sep='\t', header=True, index=True, index_label='Index', encoding='utf-8')
