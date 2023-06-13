# Noah-Manuel Michael
# Created: 06.05.2023
# Last updated: 12.05.2023
# Rescore sentences based on all possible permutations and their parse scores
# Executed on SURF Research Cloud

import pandas as pd
import spacy
from utils_parsers_correction import unique_permutations
from discodop import parser, tree


def perform_rescoring_based_on_all_possible_permutations():
	"""
	Perform rescoring on all possible unique permutations of the verb tokens for sentences of length <= 5.
	:return:
	"""
	# read in the sentences, turn them into a list
	df_test = pd.read_csv('../Fine-tuning/test_shuffled_random_all_and_verbs.tsv', sep='\t', encoding='utf-8', header=0)

	nlp = spacy.load('nl_core_news_lg')

	with open('parser_corrections_VR_max_len_5.tsv', 'w') as outfile:
		outfile.write('correct\tVR\tcorrected\tprob\n')

	for i, row in df_test.iterrows():
		sent = row['verbs_random_no_punc']
		sent_split = sent.split()

		if len(sent_split) <= 5:  # only use sentences that are up to 5 tokens in length, otherwise too many
			# permutations
			doc = nlp(sent)  # process with spacy to get the verbs
			full_sentence = []
			masked_sentence = []
			permutations = []

			for token in doc:
				full_sentence.append(token.text)
				if token.pos_ in ['VERB', 'AUX']:
					masked_sentence.append(token.text)
				else:
					masked_sentence.append('0')

			for permutation in unique_permutations(masked_sentence):  # get only unique permutations of mask
				permutation = list(permutation)
				cache = [token for x, token in enumerate(full_sentence) if token != masked_sentence[x]]  # fill mask

				for n, element in enumerate(permutation):
					if element == '0':
						permutation[n] = cache.pop(0)

				permutations.append(permutation)

			# parse all permutations
			top = 'ROOT'  # the root label in the treebank
			directory = 'nl'
			params = parser.readparam(directory + '/params.prm')
			parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
			myparser = parser.Parser(params)
			myparser.stages[-1].estimator = 'rfe'

			# store the probabilities of all parses, as well as the rest of the results and the sentences
			result_probabilities = []
			sentences_per_probability = {}
			for sent in permutations:
				result = list(myparser.parse(sent))
				for res in result:
					result_probabilities.append(res.prob)
					sentences_per_probability[f'{res.prob}'] = sent

			# get the maximum probability out of all parses
			max_prob = str(max(result_probabilities))

			# get the result and the sentence to which the maximum probability corresponds
			max_sent = sentences_per_probability[max_prob]

			# print the probability and the corresponding parse tree
			with open('parser_corrections_VR_max_len_5.tsv', 'a') as outfile:
				outfile.write(f'{row["no_punc"]}\t{row["verbs_random_no_punc"]}\t{" ".join(max_sent)}\t'
							  f'{str(max_prob)}\n')


if __name__ == '__main__':
	perform_rescoring_based_on_all_possible_permutations()
