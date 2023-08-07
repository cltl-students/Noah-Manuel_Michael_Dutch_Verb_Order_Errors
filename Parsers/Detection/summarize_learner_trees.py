# Noah-Manuel Michael
# Created: 07.08.2023
# Last updated: 07.08.2023
# simplify the learner trees

from summarize_trees import *


def simplify_trees_learner_trees():
    """
    Simplify the stree structures obtained from the disco-dop parser.
    :return: None
    """
    df = pd.read_csv(f'Data/Trees/test_Learn.tsv', encoding='utf-8', sep='\t', header=0)
    # remove verbs in path for all random shuffles
    simplified_trees = []

    for tree_string in df['tree']:
        tree = Tree.fromstring(tree_string)
        do_simplification(tree)
        tree_str = re.sub(r'\s+', ' ', str(tree))
        tree_str = re.sub(r' \)', ')', tree_str)
        simplified_trees.append(tree_str)

    df['simple_tree'] = simplified_trees
    df.to_csv(f'Data/Trees/test_Learn.tsv', encoding='utf-8', sep='\t', index=False)


if __name__ == '__main__':
    simplify_trees_learner_trees()
