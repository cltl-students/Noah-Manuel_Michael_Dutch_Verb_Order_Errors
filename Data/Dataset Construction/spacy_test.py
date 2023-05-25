import spacy
import pandas as pd

nlp = spacy.load('nl_core_news_lg')

df = pd.read_csv('Unpermuted Datasets/test.tsv', encoding='utf-8', sep='\t', header=0)

for sent in df['original']:
    doc = nlp(sent)
    for sentence in doc.sents:
        print(sentence)
        print('____')
