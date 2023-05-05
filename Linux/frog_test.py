from frog import Frog, FrogOptions
import pandas as pd


def main():
    # df = pd.read_csv('/mnt/c/Users/nwork/OneDrive/PycharmProjects/Thesis/Annotation/Data/annotated_data.tsv',
    #                  sep='\t', keep_default_na=False)
    #
    # sentence_list = df['Corrected'].tolist()

    sentence_list = []
    with open('/mnt/c/Users/nwork/OneDrive/PycharmProjects/Thesis/Annotation/Data/example_sent.txt') as infile:
        for line in infile.readlines():
            sentence_list.append(line.strip('\n'))

    print(sentence_list)

    frog = Frog(FrogOptions(parser=True))

    for sent in sentence_list:
        output = frog.process(sent)
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
    main()
