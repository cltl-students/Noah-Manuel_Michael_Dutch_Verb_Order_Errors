# Noah-Manuel Michael
# Created: 15.04.2023
# Last updated: 12.05.2023
# Clean construction of discodop trees and probabilities

from discodop import parser, tree
import pandas as pd


def get_probability_and_draw_tree_discodop():
	"""

	:return:
	"""
	df = pd.read_csv('/mnt/c/Users/nwork/OneDrive/PycharmProjects/Thesis/Annotation/Unpermuted_Datasets/leerder_annotated_data.tsv',
					 sep='\t', keep_default_na=False)
	sentence_list = df['Corrected'].tolist()

	top = 'ROOT'  # the root label in the treebank
	directory = '/home/noah/disco-dop/nl'
	params = parser.readparam(directory + '/params.prm')
	parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
	myparser = parser.Parser(params)
	myparser.stages[-1].estimator = 'rfe'

	for sent in sentence_list[:3]:
		result = list(myparser.parse(sent.split()))
		for res in result:
			print(res.prob)
			print(tree.DrawTree(res.parsetree, sent=sent.split()))
			print('________________________')

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
	get_probability_and_draw_tree_discodop()
