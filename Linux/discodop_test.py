# Noah-Manuel Michael
# 06.05.2023
# Clean construction of discodop trees and probabilities

from discodop import parser, tree
import pandas as pd


def main():
	df = pd.read_csv('/mnt/c/Users/nwork/OneDrive/PycharmProjects/Thesis/Annotation/Data/annotated_data.tsv', sep='\t',
					 keep_default_na=False)
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

	# print(result)
	# print(help(result[0]))
	# print(result[0].parsetree)
	# print(result[0].golditems)
	# print(result[0].msg)
	# print(result[0].prob)
	# print(result[0].parsetrees)


if __name__ == '__main__':
	main()
