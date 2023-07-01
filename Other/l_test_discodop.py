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
	sentence_list = ['Ik lees boeken']

	top = 'ROOT'  # the root label in the treebank
	directory = '/home/noah/disco-dop/nl'
	params = parser.readparam(directory + '/params.prm')
	parser.readgrammars(directory, params.stages, params.postagging, top=getattr(params, 'top', top))
	myparser = parser.Parser(params)
	myparser.stages[-1].estimator = 'rfe'

	for sent in sentence_list:
		result = list(myparser.parse(sent.split()))
		print(result)
		print('____________________________________________________')
		for res in result:
			print(res.name)
			print(res)
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
