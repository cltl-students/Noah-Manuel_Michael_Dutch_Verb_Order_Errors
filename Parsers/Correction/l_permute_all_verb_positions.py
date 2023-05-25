# Noah-Manuel Michael
# Created: 18.05.2023
# Last updated: 22.05.2023
# Permute only verb positions in data

import re
from frog import Frog, FrogOptions
from itertools import permutations


def permute_only_verb_positions():
    """
    Read in sentences and get all permutations of only the verb tokens' positions.
    :return: None
    """
    sentence_list = []
    with open('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Unpermuted_Datasets/Tests/parser_test_sents.txt') as infile:
        for line in infile.readlines():
            sentence_list.append(line.strip('\n'))

    print(sentence_list)

    frog = Frog(FrogOptions(parser=True))

    for sent in sentence_list:
        output = frog.process(sent)

        # sentence_length = len(output)
        # print(sentence_length)

        full_sentence = []
        masked_sentence = []

        all_permutations = []

        for token in output:
            full_sentence.append(token['text'])  # get the full tokenized sentence
            if re.findall(r'.-VP', token['chunker']):  # get only the verbs in the sentence
                masked_sentence.append(token['text'])
            else:
                masked_sentence.append(0)  # fill the rest with a placeholder

        # print(masked_sentence)
        # print(full_sentence)

        # get all possible unique permutations of the masked sentence
        permutations_of_masked_sentence = {permutation for permutation in permutations(masked_sentence)}

        # fill the masked tokens with the non-verb tokens in correct order
        for permutation in permutations_of_masked_sentence:
            cache = [token for i, token in enumerate(full_sentence) if token != masked_sentence[i]]
            permutation = list(permutation)
            for i, element in enumerate(permutation):
                if element == 0:
                    permutation[i] = cache.pop(0)

            all_permutations.append(permutation)

        print(all_permutations)

        # takes forever with long sentences


if __name__ == '__main__':
    permute_only_verb_positions()
