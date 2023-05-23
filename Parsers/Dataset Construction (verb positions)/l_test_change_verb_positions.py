# Noah-Manuel Michael
# Created: 18.05.2023
# Last updated: 18.05.2023
# Change only verb positions in data

from frog import Frog, FrogOptions
import pandas as pd


def parse_with_frog_and_get_chunks():
    """

    :return:
    """
    sentence_list = []
    with open('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Unpermuted Datasets/Tests/parser_test_sents.txt') as infile:
        for line in infile.readlines():
            sentence_list.append(line.strip('\n'))

    print(sentence_list)

    frog = Frog(FrogOptions(parser=True))

    for sent in sentence_list:
        output = frog.process(sent)
        sentence_length = len(output)
        print(sentence_length)
        print("PARSED OUTPUT=", output)
        print('_____________')
        for token in output:
            if 'WW' in token['pos']:
                print(f'{token["text"]:15}{token["chunker"]:15}{token["pos"]}')
            else:
                print(f'{token["text"]:15}{token["chunker"]:15}')

        persoonsvorm = []
        permuted_sentence = ''
        for token in output:
            if 'WW(pv' not in token['pos'] and token['text'] != '.':
                permuted_sentence = permuted_sentence + f'{token["text"]} '
            elif token['text'] == '.':
                pass
            elif len(persoonsvorm) == 0:
                persoonsvorm.append(token['text'])
            else:
                permuted_sentence = permuted_sentence + f'{token["text"]} '
        permuted_sentence = permuted_sentence + persoonsvorm[0] + '.'
        print(permuted_sentence)


if __name__ == '__main__':
    parse_with_frog_and_get_chunks()
