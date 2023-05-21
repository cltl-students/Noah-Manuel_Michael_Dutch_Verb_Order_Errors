# Noah-Manuel Michael
# Created: 15.04.2023
# Last updated: 21.05.2023
# Get the most likely parse for each sentence

import re
import pandas as pd
from discodop import parser, tree


def get_most_likely_parse():
	"""
	Parse sentences with the discodop parser. Get the most likely parse and write the sentence + its tree in string
	format to a file.
	:return: None
	"""
	sentence_list = []
	parse_list = []

	with open('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset Construction/Data/lassy_sents.txt') as infile:
		for line in infile.readlines():
			sentence_list.append(line.strip())

	sentence_list = [sent for sent in sentence_list if re.match(r'^[A-Z].*[.!?]$', sent)]  # take only sentences into
	# account that start with a capital letter and end with [.?!]

	# initiate parser
	top = 'ROOT'  # the root label in the treebank
	directory = '/home/noah/disco-dop/nl'
	params = parser.readparam(directory + '/params.prm')
	parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
	myparser = parser.Parser(params)
	myparser.stages[-1].estimator = 'rfe'

	progress_tracker = 0
	for sent in sentence_list:
		probs_of_all_results = []
		prob_to_tree = {}
		result = list(myparser.parse(sent.split()))  # parse

		for res in result:
			probs_of_all_results.append(res.prob)  # store the probability for each possible parse
			prob_to_tree[res.prob] = res.parsetree  # map the probability to the corresponding parsetree

		max_prob = max(probs_of_all_results)
		max_parse = prob_to_tree[max_prob]  # retrieve the parsetree with the highest probability
		parse_list.append(str(max_parse))

		progress_tracker += 1
		if progress_tracker % 1000 != 0:
			pass
		else:
			print(f'Sentence {progress_tracker} has been processed.')

	df = pd.DataFrame(sentence_list, columns=['original'])  # store sentences and
	df['parse'] = parse_list  # corresponding most likely parses in a df

	df.to_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset Construction/Data/lassy_sents_parsed.tsv',
			  sep='\t', index_label='index', index=True, encoding='utf-8')  # write to file

	# Available attributes: dict_keys(['name', 'parsetree', 'prob', 'parsetrees', 'fragments', 'noparse', 'elapsedtime',
    # 'numitems', 'golditems', 'totalgolditems', 'msg'])

	# print(results[0])
	# print(help(result[0]))
    # print(result[0].parsetree)
	# print(result[0].golditems)
	# print(result[0].msg)
	# print(result[0].prob)
	# print(result[0].parsetrees)
	# print(tree.DrawTree(result[0].parsetree, sent=sent.split())


if __name__ == '__main__':
	get_most_likely_parse()
