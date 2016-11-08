import random
import numpy as np

def flatten_list(unflattened_list):
    """Flattens a list of lists.

    Args:
        unflattened_list (list): A list of lists.

    Returns:
        list: The concatenation of all sublists of unflattened_list.
    """
    return [item for sublist in unflattened_list for item in sublist]

def randomize_data(X, Y):
    """Shuffles two lists based on the same random seed.

    Given two lists of equal length, generates a random re-ordering of indices
    and reorders both lists by this same random ordering.

    Args:
        X (list): A list of length N.
        Y (list): A list of length N.

    Returns:
        list: The list X having been reordered by the random shuffle.
        list: The list Y having been reordered by the same random shuffle.
        list: The random shuffle itself, for use in accuracy testing later on.
    """
    random_ordering = np.random.choice(range(len(X)), len(X), False)
    return X[random_ordering, ], Y[random_ordering], random_ordering