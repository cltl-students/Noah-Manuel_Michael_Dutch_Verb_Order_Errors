# Noah-Manuel Michael
# Created: 15.04.2023
# Last updated: 24.05.2023
# Get the most likely parse for each sentence

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
	df_train = pd.read_csv('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset Construction/'
						   'Unpermuted Datasets/train.tsv', sep='\t', encoding='utf-8', header=0)

	sentence_list = df_train['original'].tolist()
	parse_list = []

	# initiate parser
	top = 'ROOT'  # the root label in the treebank
	directory = '/home/noah/disco-dop/nl'
	params = parser.readparam(directory + '/params.prm')
	parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
	myparser = parser.Parser(params)
	myparser.stages[-1].estimator = 'rfe'

	# begin = int()
	# if 'lassy_sents_parsed.tsv' not in os.listdir('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset '
	# 											  'Construction/Data/'):
	# 	with open('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset Construction/Data/lassy_sents_parsed.tsv',
	# 		  	  'w', encoding='utf-8') as outfile:
	# 		outfile.write('index\toriginal\tparse\n')
	# else:
	# 	with open('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Data/Dataset Construction/Data/lassy_sents_parsed.tsv',
	# 		  	  encoding='utf-8') as infile:
	# 		for line in infile.readlines():
	# 			begin = line.split()[0]

	for i, sent in enumerate(sentence_list):
	# for i, sent in enumerate(sentence_list[int(begin)+1:]):
	# 	if len(sent.split()) >= 30:  # too long sentences can't be processed
	# 		continue
	# 	i = i+int(begin)+1
		probs_of_all_results = []
		prob_to_tree = {}
		result = list(myparser.parse(sent.split()))  # parse

		for res in result:
			probs_of_all_results.append(res.prob)  # store the probability for each possible parse
			prob_to_tree[res.prob] = res.parsetree  # map the probability to the corresponding parsetree

		max_prob = max(probs_of_all_results)
		max_parse = prob_to_tree[max_prob]  # retrieve the parsetree with the highest probability
		parse_list.append(str(max_parse))

		with open('/mnt/c/Users/nwork/OneDrive/Studium/ma_thesis/Parsers/Detection/train_parsed.tsv', 'a',
				  encoding='utf-8') as outfile:
			outfile.write(f'{i}\t{sent}\t{str(max_parse)}\n')

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
