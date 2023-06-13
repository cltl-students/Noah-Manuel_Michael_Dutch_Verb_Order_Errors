# Noah-Manuel Michael
# Created: 13.06.2023
# Last updated: 13.06.2023
# utils for correction with parsers
# This script was pair-programmed with Chat-GPT (v4)


def unique_permutations(seq):
    """
    Generator object that yields all unique permutations for a sequence. Elements of the sequence are treated as unique
    based on their value, not their position (as opposed to the itertools functions).
    :param seq:
    :return:
    """
    seq = sorted(seq)  # Ensure the sequence is in a sorted order
    yield tuple(seq)

    while True:
        for i in reversed(range(len(seq) - 1)):
            if seq[i] < seq[i + 1]:  # Find the smallest 'i' such that seq[i] < seq[i + 1]
                break
        else:  # No such 'i' means the sequence is in descending order, we're done
            return

        # Find the largest 'j' such that j > i and seq[j] > seq[i]
        for j in reversed(range(i + 1, len(seq))):
            if seq[j] > seq[i]:
                break

        # Swap seq[i] and seq[j]
        seq[i], seq[j] = seq[j], seq[i]

        # Reverse the elements at position i+1 till end
        seq[i + 1:] = reversed(seq[i + 1:])

        yield tuple(seq)
