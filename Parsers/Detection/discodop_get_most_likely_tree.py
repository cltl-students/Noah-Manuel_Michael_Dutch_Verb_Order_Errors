# Noah-Manuel Michael
# Created: 15.04.2023
# Last updated: 27.05.2023
# Get the most likely parse for each sentence
# Script executed on SURF Research Cloud

import re
import os
import pandas as pd
from discodop import parser, tree


def get_most_likely_parse():
	"""
	Parse sentences with the discodop parser. Get the most likely parse and write the sentence + its tree in string
	format to a file.
	:return: None
	"""
	for split in ['test', 'dev', 'train']:
		df = pd.read_csv(f'../{split}.tsv', sep='\t', encoding='utf-8', header=0)  # SURF

		parse_list = []

		# initiate parser
		top = 'ROOT'  # the root label in the treebank
		directory = '/home/nmichael/disco-dop/nl'  # SURF
		params = parser.readparam(directory + '/params.prm')
		parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
		myparser = parser.Parser(params)
		myparser.stages[-1].estimator = 'rfe'

		with open(f'../{split}_correct_parsed.tsv', 'w', encoding='utf-8') as outfile:  # SURF
			outfile.write('index\tsentence\ttree\n')

		for i, sent in enumerate(df['spaced']):
			probs_of_all_results = []
			prob_to_tree = {}
			result = list(myparser.parse(sent.split()))  # parse

			for res in result:
				probs_of_all_results.append(res.prob)  # store the probability for each possible parse
				prob_to_tree[res.prob] = res.parsetree  # map the probability to the corresponding parsetree

			max_prob = max(probs_of_all_results)
			max_parse = prob_to_tree[max_prob]  # retrieve the parsetree with the highest probability
			parse_list.append(str(max_parse))

			with open(f'../{split}_correct_parsed.tsv', 'a', encoding='utf-8') as outfile:  # SURF
				outfile.write(f'{i}\t{sent}\t{str(max_parse)}\n')


if __name__ == '__main__':
	get_most_likely_parse()
