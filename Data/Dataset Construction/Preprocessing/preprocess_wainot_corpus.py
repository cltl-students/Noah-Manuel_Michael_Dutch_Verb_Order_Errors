# Noah-Manuel Michael
# Created: 23.05.2023
# Last updated: 23.05.2023
# Preprocess the wainot data
# This script was pair-programmed with ChatGPT (v4)

import glob
import spacy
import re
import pandas as pd
from bs4 import BeautifulSoup


def preprocess_wainot_corpus():
    wainot_data = []

    file_list = glob.glob('../Unpermuted Datasets/wainot_corpus/*.xml')

    # Read the XML files
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as infile:
            xml_data = infile.read()

        # Parse the XML
        soup = BeautifulSoup(xml_data, 'xml')

        # Find all the <p> tags
        p_tags = soup.find_all('p')

        # Iterate over the <p> tags
        for p_tag in p_tags:
            # Get the text content within the <p> tag
            p_text = p_tag.get_text(separator=' ').strip()
            if p_text != '':
                wainot_data.append(p_text.strip())

    # use spacy to perform sent tokenize
    nlp = spacy.load('nl_core_news_sm')

    wainot_sents = []

    for text in wainot_data:
        doc = nlp(text)
        for sent in doc.sents:
            if len(sent.text) > 10 and re.match(r'^[A-Z].*[.!?]$', sent.text):
                sent_text = re.sub(r' ', ' ', sent.text)
                sent_text = re.sub(r'  ', ' ', sent_text)
                wainot_sents.append(sent_text)

    df = pd.DataFrame(wainot_sents, columns=['original'])

    df.to_csv('../Unpermuted Datasets/wainot_data.tsv', sep='\t', encoding='utf-8', index_label='index')


if __name__ == '__main__':
    preprocess_wainot_corpus()
