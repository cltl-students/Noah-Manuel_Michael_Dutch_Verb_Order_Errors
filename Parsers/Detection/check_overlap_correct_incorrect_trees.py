# Noah-Manuel Michael
# Created: 08.06.2023
# Last updated: 11.06.2023
# Check how much overlap there is in tree structures in the correct and incorrect training data

import pandas as pd


def get_overlap_statistics():
    df_train_c = pd.read_csv('Data/Trees/train_Correct.tsv', encoding='utf-8', sep='\t', header=0)
    df_train_ar = pd.read_csv('Data/Trees/train_Rand.tsv', encoding='utf-8', sep='\t', header=0)
    df_train_vr = pd.read_csv('Data/Trees/train_Verbs.tsv', encoding='utf-8', sep='\t', header=0)

    unique_trees_c = {tree for tree in df_train_c['tree']}
    unique_trees_ar = {tree for tree in df_train_ar['tree']}
    unique_trees_vr = {tree for tree in df_train_vr['tree']}

    unique_simple_trees_c = {tree for tree in df_train_c['simple_tree']}
    unique_simple_trees_ar = {tree for tree in df_train_ar['simple_tree']}
    unique_simple_trees_vr = {tree for tree in df_train_vr['simple_tree']}

    print(f'Unique trees in correct train data: {len(unique_trees_c)}')
    print(f'Unique trees in AR train data: {len(unique_trees_ar)}')
    print(f'Unique trees in VR train data: {len(unique_trees_vr)}')

    print(f'Overlap in correct and AR: {len(unique_trees_c.intersection(unique_trees_ar))}')
    print(f'Overlap in correct and VR: {len(unique_trees_c.intersection(unique_trees_vr))}')

    # print(unique_trees_c.intersection(unique_trees_ar))
    # print(unique_trees_c.intersection(unique_trees_vr))

    print(f'Unique simple trees in correct train data: {len(unique_simple_trees_c)}')
    print(f'Unique simple trees in AR train data: {len(unique_simple_trees_ar)}')
    print(f'Unique simple trees in VR train data: {len(unique_simple_trees_vr)}')

    print(f'Overlap in simple correct and AR: {len(unique_simple_trees_c.intersection(unique_simple_trees_ar))}')
    print(f'Overlap in simple correct and VR: {len(unique_simple_trees_c.intersection(unique_simple_trees_vr))}')

    # print(unique_simple_trees_c.intersection(unique_simple_trees_ar))
    # print(unique_simple_trees_c.intersection(unique_simple_trees_vr))


if __name__ == '__main__':
    get_overlap_statistics()
