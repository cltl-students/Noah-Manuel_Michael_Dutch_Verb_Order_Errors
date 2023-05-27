# Noah-Manuel Michael
# Created: 26.05.2023
# Last updated: 26.05.2023
# Inspect how spaCy handles punctuation when tokenizing

import spacy

list_of_sents = ['Dit is een zin.', 'Dit is een zin .', 'Hier zijn meer dan 85.000 mensen.',
                 'Hier zijn meer dan 85.000 mensen .', 'Ik ben groot, omdat mijn moeder dat zegt.',
                 'Ik ben groot , omdat mijn moeder dat zegt .', 'Hij zei van: "Hé, dit is mijn boek."',
                 'Hij zei van : " Hé , dit is mijn boek . "', 'Hij zei van:"Hé, dit is mijn boek."']

nlp = spacy.load('nl_core_news_lg')

for sent in list_of_sents:
    doc = nlp(sent)
    print(len(doc))
    print(doc)
    for token in doc:
        print(token)
    print('______')
