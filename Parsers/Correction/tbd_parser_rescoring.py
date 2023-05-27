# Noah-Manuel Michael
# Created: 06.05.2023
# Last updated: 12.05.2023
# Rescore sentences based on all possible permutations and their parse scores

import itertools
import re
from discodop import parser, tree


def load_sentences():
	"""

	:return:
	"""
	...


def perform_rescoring_based_on_all_possible_permutations():
	"""

	:return:
	"""
	# read in the sentences, turn them into a list, punctuation at the end of the sentence is its own token
	with open('/mnt/c/Users/nwork/OneDrive/PycharmProjects/Thesis/Annotation/Data/short_story.txt') as infile:
		sentences = []
		for line in infile.readlines():
			if line.endswith('.\n'):
				normalized_line = re.sub(r'\.\n', ' .', line)
			elif line.endswith('?\n'):
				normalized_line = re.sub(r'\?\n', ' ?', line)
			elif line.endswith('!\n'):
				normalized_line = re.sub(r'!\n', ' !', line)
			sentences.append(normalized_line.split())

	# get all possible permutations of the sentence
	all_permutations = []
	permutations = itertools.permutations(sentences[-1])
	for p in permutations:
		list_of_p = list(p)  # note: later add another shuffle method where the punctuation is always at the end of the
		# sentence
		all_permutations.append(list_of_p)

	# parse all permutations
	top = 'ROOT'  # the root label in the treebank
	directory = '/home/noah/disco-dop/nl'
	params = parser.readparam(directory + '/params.prm')
	parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
	myparser = parser.Parser(params)
	myparser.stages[-1].estimator = 'rfe'

	# store the probabilities of all parses, as well as the rest of the results and the sentences
	result_probabilities = []
	results_per_probability = {}
	sentences_per_probability = {}
	for sent in all_permutations:
		result = list(myparser.parse(sent))
		for res in result:
			result_probabilities.append(res.prob)
			results_per_probability[f'{res.prob}'] = res
			sentences_per_probability[f'{res.prob}'] = sent
			# print(res.prob)
			# print(tree.DrawTree(res.parsetree, sent=sent))
			# print('________________________')

	# get the maximum probability out of all parses
	max_prob = str(max(result_probabilities))

	# get the result and the sentence to which the maximum probability corresponds
	max_result = results_per_probability[max_prob]
	max_sent = sentences_per_probability[max_prob]

	# print the probability and the corresponding parse tree
	print(max_prob)
	print(tree.DrawTree(max_result.parsetree, sent=max_sent))

	# print(result)
	# print(help(result[0]))
	# print(result[0].parsetree)
	# print(result[0].golditems)
	# print(result[0].msg)
	# print(result[0].prob)
	# print(result[0].parsetrees)


if __name__ == '__main__':
	perform_rescoring_based_on_all_possible_permutations()
