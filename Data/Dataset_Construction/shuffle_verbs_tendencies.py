# Noah-Manuel Michael
# Created: 19.06.2023
# Last updated: 23.06.2023
# Permute only verb positions in data for dataset creation; get only a likely permutation according to tendencies

import pandas as pd
import spacy
from collections import defaultdict


def get_shuffled_dataset_according_to_learner_tendencies(filepath):
    """
    Read in test data and permute verb positions according to learner tendencies.
    :param filepath: path to test file
    :return: None
    """
    class NotAValidSubclause(Exception):
        pass

    class NoDifferenceInSubClause(Exception):
        pass

    class NoDifferenceInMainClause(Exception):
        pass

    df = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
    nlp = spacy.load('nl_core_news_lg')

    desired_distribution = {'mainmedialaftersubj': 329, 'mainfinal': 164, 'mainmedialbeforesubj': 112,
                            'mainfinalbeforenonfinite': 75, 'maininitial': 67, 'mainrandom': 374, 'maincorrect': 10223,
                            'submedialaftersubj': 433, 'subinitial': 232, 'submedialbeforesubj': 82, 'subrandom': 374,
                            'subcorrect': 1121}
    actual_distibution = defaultdict(int)

    VT = []
    label_general = []
    label_detailed = []

    for i, row in df.iterrows():
        doc = nlp(row['no_punc'])
        full_sent = [token.text for token in doc]
        copy_of_full_sent = full_sent.copy()
        all_dep = [token.dep_ for token in doc]
        all_morph = [str(token.morph) if str(token.morph) != '' else '0' for token in doc]  # morphological information

        try:
            # check for subordinate clauses
            if 'mark' in all_dep:
                beginning_subclause: int = all_dep.index('mark')

                # check if the subordinate clause ends at some point or if it is the last clause in the sentence
                if 'cc' not in all_dep[beginning_subclause:]:
                    limit_subclause: None = None  # assigning none equals not assigning any limit
                else:  # conjunction, the subordinate clause must be followed by another clause
                    limit_subclause: int = all_dep.index('cc')

                # check if there is a finite verb in the subordinate clause
                finite_verb_positions = [i for i, s in enumerate(all_morph[beginning_subclause:limit_subclause])
                                         if 'VerbForm=Fin' in s]
                if finite_verb_positions:
                    finite_verb_in_subclause_at = finite_verb_positions[0]
                    finite_verb_of_subclause_at_absolute = finite_verb_in_subclause_at + beginning_subclause

                    # check if there is a subject in the subordinate clause
                    if 'nsubj' in all_dep[beginning_subclause:limit_subclause]:
                        subject_of_subclause_at_absolute: int = \
                            all_dep[beginning_subclause:limit_subclause].index('nsubj') + beginning_subclause

                        try:
                            # subclause, initial, finite verb in first position
                            if actual_distibution['subinitial'] < desired_distribution['subinitial']:
                                finite_verb = full_sent.pop(finite_verb_of_subclause_at_absolute)
                                full_sent.insert(beginning_subclause + 1, finite_verb)

                                if copy_of_full_sent != full_sent:  # make sure there is actually a difference to the
                                    # original sentence
                                    actual_distibution['subinitial'] += 1
                                    VT.append(' '.join(full_sent))
                                    label_general.append('verb-final')
                                    label_detailed.append('subinitial')
                                    continue
                                else:
                                    raise NoDifferenceInSubClause

                            # subclause, medial, finite verb after subject
                            if actual_distibution['submedialaftersubj'] < desired_distribution['submedialaftersubj']:
                                finite_verb = full_sent.pop(finite_verb_of_subclause_at_absolute)
                                if finite_verb_of_subclause_at_absolute < subject_of_subclause_at_absolute:
                                    full_sent.insert(subject_of_subclause_at_absolute, finite_verb)
                                else:
                                    full_sent.insert(subject_of_subclause_at_absolute + 1, finite_verb)

                                if copy_of_full_sent != full_sent:
                                    actual_distibution['submedialaftersubj'] += 1
                                    VT.append(' '.join(full_sent))
                                    label_general.append('verb-final')
                                    label_detailed.append('submedialaftersubj')
                                    continue
                                else:
                                    raise NoDifferenceInSubClause

                            # subclause, medial, finite verb before subject
                            elif actual_distibution['submedialbeforesubj'] < \
                                    desired_distribution['submedialbeforesubj']:
                                finite_verb = full_sent.pop(finite_verb_of_subclause_at_absolute)
                                if finite_verb_of_subclause_at_absolute < subject_of_subclause_at_absolute:
                                    full_sent.insert(subject_of_subclause_at_absolute - 1, finite_verb)
                                else:
                                    full_sent.insert(subject_of_subclause_at_absolute, finite_verb)

                                if copy_of_full_sent != full_sent:
                                    actual_distibution['submedialbeforesubj'] += 1
                                    VT.append(' '.join(full_sent))
                                    label_general.append('verb-final')
                                    label_detailed.append('submedialbeforesubj')
                                    continue
                                else:
                                    raise NoDifferenceInSubClause

                            else:
                                raise NoDifferenceInSubClause

                        except NoDifferenceInSubClause:
                            if actual_distibution['subrandom'] < desired_distribution['subrandom']:
                                actual_distibution['subrandom'] += 1
                                VT.append(row['verbs_random_no_punc'])
                                label_general.append('verb-final')
                                label_detailed.append('subrandom')
                                continue
                            elif actual_distibution['subcorrect'] < desired_distribution['subcorrect']:
                                actual_distibution['subcorrect'] += 1
                                VT.append(row['no_punc'])
                                label_general.append('correct')
                                label_detailed.append('correct')
                                continue

                    else:  # no subject
                        raise NotAValidSubclause
                else:  # no finite verb
                    raise NotAValidSubclause
            else:  # sentence must be main clause
                raise NotAValidSubclause

        except NotAValidSubclause:
            # check if main clause is final clause
            if 'cc' not in all_dep:
                limit_mainclause: None = None
            else:
                limit_mainclause: int = all_dep.index('cc')

            # check if there is a finite verb in the main clause
            finite_verb_positions = [i for i, s in enumerate(all_morph[:limit_mainclause])
                                     if 'VerbForm=Fin' in s]

            try:
                if finite_verb_positions:
                    finite_verb_of_mainclause_at_absolute: int = finite_verb_positions[0]

                    # check if there is a subject in the main clause
                    if 'nsubj' in all_dep[:limit_mainclause]:
                        subject_of_mainclause_at_absolute: int = all_dep[:limit_mainclause].index('nsubj')

                        # main clause, medial, finite verb after subject
                        if actual_distibution['mainmedialaftersubj'] < desired_distribution['mainmedialaftersubj']:
                            finite_verb = full_sent.pop(finite_verb_of_mainclause_at_absolute)
                            if finite_verb_of_mainclause_at_absolute < subject_of_mainclause_at_absolute:
                                full_sent.insert(subject_of_mainclause_at_absolute, finite_verb)
                            else:
                                full_sent.insert(subject_of_mainclause_at_absolute + 1, finite_verb)

                            if copy_of_full_sent != full_sent:
                                actual_distibution['mainmedialaftersubj'] += 1
                                VT.append(' '.join(full_sent))
                                label_general.append('verb-second')
                                label_detailed.append('mainmedialaftersubj')
                                continue
                            else:
                                raise NoDifferenceInMainClause

                        # main clause, medial, finite verb before subject
                        elif actual_distibution['mainmedialbeforesubj'] < desired_distribution['mainmedialbeforesubj']:
                            finite_verb = full_sent.pop(finite_verb_of_mainclause_at_absolute)
                            if finite_verb_of_mainclause_at_absolute < subject_of_mainclause_at_absolute:
                                full_sent.insert(subject_of_mainclause_at_absolute - 1, finite_verb)
                            else:
                                full_sent.insert(subject_of_mainclause_at_absolute, finite_verb)

                            if copy_of_full_sent != full_sent:
                                actual_distibution['mainmedialbeforesubj'] += 1
                                VT.append(' '.join(full_sent))
                                label_general.append('verb-second')
                                label_detailed.append('mainmedialbeforesubj')
                                continue
                            else:
                                raise NoDifferenceInMainClause

                    else:
                        raise NoDifferenceInMainClause

                    # check if there is a non-finite verb in the main clause
                    non_finite_verb_positions = [i for i, s in enumerate(all_morph[:limit_mainclause])
                                                 if 'VerbForm=Inf' in s]
                    if non_finite_verb_positions:
                        non_finite_verb_in_mainclause_at_absolute: int = non_finite_verb_positions[0]

                        # main clause, final, finite verb before non-finite
                        if actual_distibution['mainfinalbeforenonfinite'] < \
                                desired_distribution['mainfinalbeforenonfinite']:
                            finite_verb = full_sent.pop(finite_verb_of_mainclause_at_absolute)
                            if finite_verb_of_mainclause_at_absolute < non_finite_verb_in_mainclause_at_absolute:
                                full_sent.insert(non_finite_verb_in_mainclause_at_absolute - 1, finite_verb)
                            else:
                                full_sent.insert(non_finite_verb_in_mainclause_at_absolute, finite_verb)

                            if copy_of_full_sent != full_sent:
                                actual_distibution['mainfinalbeforenonfinite'] += 1
                                VT.append(' '.join(full_sent))
                                label_general.append('verb-second')
                                label_detailed.append('mainfinalbeforenonfinite')
                                continue
                            else:
                                raise NoDifferenceInMainClause

                    else:
                        raise NoDifferenceInMainClause

                    # main clause, initial, finite verb in first position
                    if actual_distibution['maininitial'] < desired_distribution['maininitial']:
                        finite_verb = full_sent.pop(finite_verb_of_mainclause_at_absolute)
                        full_sent.insert(0, finite_verb)

                        if copy_of_full_sent != full_sent:
                            actual_distibution['maininitial'] += 1
                            VT.append(' '.join(full_sent))
                            label_general.append('verb-second')
                            label_detailed.append('maininitial')
                            continue
                        else:
                            raise NoDifferenceInMainClause

                    # main clause, final, finite verb in final position
                    elif actual_distibution['mainfinal'] < desired_distribution['mainfinal']:
                        finite_verb = full_sent.pop(finite_verb_of_mainclause_at_absolute)

                        if limit_mainclause:
                            full_sent.insert(limit_mainclause + 1, finite_verb)
                        else:
                            full_sent.append(finite_verb)

                        if copy_of_full_sent != full_sent:
                            actual_distibution['mainfinal'] += 1
                            VT.append(' '.join(full_sent))
                            label_general.append('verb-second')
                            label_detailed.append('mainfinal')
                            continue
                        else:
                            raise NoDifferenceInMainClause

                    else:
                        raise NoDifferenceInMainClause

                else:
                    raise NoDifferenceInMainClause

            except NoDifferenceInMainClause:
                if actual_distibution['mainrandom'] < desired_distribution['mainrandom']:
                    actual_distibution['mainrandom'] += 1
                    VT.append(row['verbs_random_no_punc'])
                    label_general.append('verb-second')
                    label_detailed.append('mainrandom')
                    continue
                elif actual_distibution['maincorrect'] < desired_distribution['maincorrect']:
                    actual_distibution['maincorrect'] += 1
                    VT.append(row['no_punc'])
                    label_general.append('correct')
                    label_detailed.append('correct')
                    continue

    print(actual_distibution)

    df['tendencies_no_punc'] = VT
    df['general_error_label'] = label_general
    df['detailed_error_label'] = label_detailed

    df.to_csv(f'{filepath.strip(".tsv")}_and_tendencies.tsv', header=True, index=False, encoding='utf-8', sep='\t')


if __name__ == '__main__':
    filepath = 'Permuted_Datasets/test_shuffled_random_all_and_verbs.tsv'
    get_shuffled_dataset_according_to_learner_tendencies(filepath)
