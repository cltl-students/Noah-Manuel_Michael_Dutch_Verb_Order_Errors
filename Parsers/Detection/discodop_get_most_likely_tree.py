# Noah-Manuel Michael
# Created: 15.04.2023
# Last updated: 01.06.2023
# Get the most likely parse for each sentence
# Script executed on SURF Research Cloud
# I only parse the test set and 80000 sents from the train because of how long it takes to parse the sentences

import pandas as pd
from discodop import parser, tree


def get_most_likely_parse(split, datatype):
	"""
	Parse sentences with the discodop parser. Get the most likely parse and write the sentence + its tree in string
	format to a file.
	:param str split: train, test
	:param str datatype: no_punc, scrambled_no_punc, verbs_random_no_punc
	:return: None
	"""
	if datatype == 'no_punc':
		correctness = 'correct'
	else:
		correctness = 'incorrect'

	if split == 'test':
		sample = ''
	else:
		sample = '_sampled'

	df = pd.read_csv(f'../{split}_shuffled_random_all_and_verbs{sample}.tsv', sep='\t', encoding='utf-8', header=0)

	parse_list = []

	# initiate parser
	top = 'ROOT'  # the root label in the treebank
	directory = '/home/nmichael/disco-dop/nl'  # SURF
	params = parser.readparam(directory + '/params.prm')
	parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
	myparser = parser.Parser(params)
	myparser.stages[-1].estimator = 'rfe'

	if datatype == 'verbs_random_no_punc':
		with open(f'../{split}_{correctness}_verbs_parsed.tsv', 'w', encoding='utf-8') as outfile:  # SURF
			outfile.write('index\tsentence\ttree\n')
	else:
		with open(f'../{split}_{correctness}_parsed.tsv', 'w', encoding='utf-8') as outfile:  # SURF
			outfile.write('index\tsentence\ttree\n')

	for i, sent in enumerate(df[datatype]):
		probs_of_all_results = []
		prob_to_tree = {}
		result = list(myparser.parse(sent.split()))  # parse

		for res in result:
			probs_of_all_results.append(res.prob)  # store the probability for each possible parse
			prob_to_tree[res.prob] = res.parsetree  # map the probability to the corresponding parsetree

		max_prob = max(probs_of_all_results)
		max_parse = prob_to_tree[max_prob]  # retrieve the parsetree with the highest probability
		parse_list.append(str(max_parse))

		if datatype == 'verbs_random_no_punc':
			with open(f'../{split}_{correctness}_verbs_parsed.tsv', 'a', encoding='utf-8') as outfile:  # SURF
				outfile.write(f'{i}\t{sent}\t{str(max_parse)}\n')
		else:
			with open(f'../{split}_{correctness}_parsed.tsv', 'a', encoding='utf-8') as outfile:  # SURF
				outfile.write(f'{i}\t{sent}\t{str(max_parse)}\n')


if __name__ == '__main__':
	for split in ['test', 'train']:
		for datatype in ['no_punc', 'scrambled_no_punc', 'verbs_random_no_punc']:
			get_most_likely_parse(split, datatype)
