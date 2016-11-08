"""
This file implements the framework outlined in our paper.

To run: 
python cluster.py --authors=<author_list> --pos=<boolean> --k=<num_clusters> \
    --randomize=<boolean> --num_runs=<integer> --ngram_range=<tuple>

--authors: A comma seperated list of directories containing documents to be 
    clustered. Names must match folders in ./data/ containing documents. 
    Required argument.

--pos: If True, will take data from ./data/<author_name>_pos/ instead of
    ./data/<author_name>. ./data/<author_name>_pos/ should contain 
    'POS-translated' documents. Default: True

--k: Number of clusters that will be produced. If k < len(authors), then all
    len(authors) choose k combinations will be tested. Default: len(authors)

--randomize: If True, document order will be randomly shuffled before fed 
    through the framework. If False, design matrix will contain documents 
    ordered by true clusters (even though the framework gains no knowledge from 
    this ordering a priori). Default: True

--num_runs: The number of times to perform clustering and average results over.
    Default: 10

--ngram_range: Tuple of the form (integer, integer) that specifies the range of
    ngrams to include as features. Default: (3, 5)

Example:
python cluster.py --authors='dowd, krugman' --pos=True --k=2 --randomize=True \
    --num_runs=10
"""

import itertools, operator, os, re, random, sys, util
from collections import Counter
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from scipy.spatial import distance
from util import flatten_list, randomize_data


from scipy.spatial.distance import cosine

""" Generic-object constructors"""
def random_forest():
    return RandomForestClassifier(n_estimators=500)


def spectral(X, n):
    instance = SpectralClustering(n_clusters=n, affinity='linear')
    return instance.fit_predict(X)


""" Helper functions for main cluster algorithm"""
def auth_paths(auth, POS):
    """Returns list of filepaths to documents.

    Assumes that files are stored in the following way.
    './data/{author name (+ _pos)}/{files}'

    Args:
        auth (str): A string name of an author for whom you have documents.
        POS (bool): A boolean value representing whether you want the original
            document or the POS-converted document.

    Returns:
        list: The filepath of documents belonging to the author 
        or the filepath of POS-converted documents, depending on POS.
    """
    path = './data/{0}_pos/' if POS else './data/{0}/'
    return [(path + '{1}').format(auth, i)
            for i in os.listdir(path.format(auth)) if i != '.DS_Store']


def get_matrix(authors, POS, ngram_range):
    """Builds design matrix and runs TFIDF.

    Given a list of authors, builds a design matrix on n-gram
    feature sets over all documents (either POS or words) for
    all authors.

    Args:
        authors (list): A list of strings of Author names.
        POS (boolean): A boolean value representing whether you want the 
            original documents or the POS-converted documents.

    Returns:
        numpy.ndarray: The vectorized documents fetched by auth_paths.
                M = Number of Features
                N = Number of documents
            The matrix value at (n, m) represents the number of occurences
            of the mth feature (n-gram) in the nth documents, re-weighted
            by TDIDF.
    """
    paths = [auth_paths(i, POS) for i in authors]
    vectorizer = TfidfVectorizer(input='filename', ngram_range=ngram_range)
    return vectorizer.fit_transform(flatten_list(paths)).toarray()


def get_labels(authors, POS):
    """Returns true labels for list of authors.

    Given a list of authors, returns a list of integers representing
    the documents for all documents written by an author in your author
    list. These integers represent 'true labels'.

    Args:
        authors (list): A list of strings of Author names.
        POS (boolean): A boolean value representing whether you want the 
            original document or the POS-converted document.

    Returns:
        list: The length of this list is equal to the total
        number of documents written by authors in your authors list. These
        integers are subject to the requirement that any two documents written
        by the same author will be represented by the same integer.
    """
    author_lengths = [len(i)
                      for i in [auth_paths(auth, POS) for auth in authors]]
    unflattened_labels = [author_length * [i]
                          for author_length, i in [(author_lengths[
                              i], i) for i in range(len(author_lengths))]]
    return flatten_list(unflattened_labels)


def cluster_accuracy(predicted_labels, true_labels, n):
    """Returns the F1 Score of the predicted_labels.

    Given the predicted labels from the classifier, calculates the F1 Score
    with regards to the correct true authors under 'micro' weighting. 

    Args:
        predicted_labels (list): The labels assigned to each document by 
            the classifer.
        true_labels (list): The true labels assigned to each document by 
            true_labels.
        n (int): The number of authors we are considering.

    Returns:
        (float): The F1 score of the predicted labels versus the
        true labels. F1 is calculated as documented here:
            http://scikit-learn.org/stable/modules/model_evaluation.html
    """
    perms = itertools.permutations(range(len(set(true_labels))))
    result = [
        f1_score(predicted_labels,
                 [perm[label] for label in true_labels],
                 pos_label=None,
                 labels=range(n),
                 average='weighted') for perm in perms
    ]
    return max(result)


def get_core_indexes(matrix, cluster_indexes, n):
    """Finds core documents based on cosine similarity.

    Given a design matrix and a subset of documents in that matrix which
    were clustered together, returns a list of
    documents in that subset which represent "core" elements of that cluster.

    Args:
        matrix (numpy.ndarray): A design matrix representing features of each 
            document.
        cluster_indexes (list): A list of integers representing the indices of
            documents in the feature matrix which were clustered together.
        n (integer): The number of authors being compared.

    Returns:
        list: The indices of documents which
        represent the core elements of the given cluster.
    """
    cluster_mean = np.mean(matrix[cluster_indexes], axis=0)
    angles = [distance.euclidean(matrix[i, ], cluster_mean)
              for i in cluster_indexes]
    return [cluster_indexes[i]
            for i in range(len(cluster_indexes))
            if np.mean(angles) - 2 * np.std(angles) < angles[i] and angles[i] <
            np.mean(angles) + 2 * np.std(angles)]


""" Primary functions for Cluster algorithm."""


def supervised_improvement(matrix, cluster_cores):
    """Classifies the documents based on core elements.

    Clusters the documents represented by the design matrix using core elements
    listed in cluster_cores.

    Args:
        matrix (numpy.ndarray): A design matrix representing features of each 
            document.
        cluster_cores (list): A list of cluster cores to be used by
            the classifier.
    Returns:
        list: The predicted label for each document by a Random Forest 
            classifier, having been trained on the cluster cores found from 
            Spectral Clustering.
    """
    y = flatten_list([[i] * len(cluster_cores[i])
                      for i in range(len(cluster_cores))])
    matrix_trained = np.vstack([matrix[core] for core in cluster_cores])

    clf = random_forest()
    clf.fit(matrix_trained, y)
    return clf.predict(matrix)


def cluster(num_authors, POS, matrix):
    """Predicts labels of documents from authors.

    Given a design matrix of featurized documents, clusters the documents using 
    Spectral Clustering, then calculates core elements of each cluster and 
    re-classifies the documents based on the core labels using a Random Forest 
    Classifier.

    Args:
        num_authors (integer): The number of authors represented in our 
            documents.
        POS (boolean): A boolean value representing whether you want the 
            original document or the POS-converted document.
        matrix (np.ndarray): The design matrix for your documents.

    Returns:
        list: The predicted labels of each document as an integer list,
                as first determined by Spectral Clustering and then refined
                by our classifer.
    """
    predicted_labels = spectral(matrix, num_authors)
    cluster_labels = [[i for i, x in enumerate(predicted_labels) if x == j]
                      for j in range(num_authors)]
    cluster_cores = [get_core_indexes(matrix, i, num_authors)
                     for i in cluster_labels]
    return supervised_improvement(matrix, cluster_cores)


def authorial_decomposition(authors, POS, k, randomize, num_runs, ngram_range):
    """Clusters documents by author and calculates accuracy of clustering.

    Given a list of authors, runs our clustering algorithm on all documents 
    written by any of the given authors. Then calculates the accuracy of our 
    clustering based on true labels given by the file structure. Runs this 
    algorithm num_runs times, then returns the average over all runs.

    Alternatively, if k is specified, runs authorial_decomposition (with k=None)
    on all k-length combinations of authors and returns a list representing the 
    result of each run.

    Args:
        authors (list): A list of strings of Author names.
        POS (boolean): A boolean value representing whether you want the 
            original document or the POS-converted document. This defaults to 
            True, which fetches the POS-converted documents.
        k (integer): This defaults to None, which has no effect. If k is 
            specified, this determines the length of the combinations in the 
            alternate use of this function.
        randomize (boolean): Whether or not to randomize the order of the 
            documents. Setting randomize to False is useful for testing
            purposes.
        num_runs (integer): Number of runs to conduct of which results are 
            averaged over
        ngram_range (tuple): Tuple of the form (integer, interger) that 
            specifies the range of ngrams to include as features

    Returns:
        This function returns one of two types.

        If k is not specified (or specified as None):
            float: The F1 accuracy of our clustering algorithm over the 
                   authors given by the authors in the authors input. For 
                   example: 
                   authorial_decomposition(['friedman', 'krugman', 'dowd']) 
                        = 0.907

        Alternatively, if k is specified: 
            list: the accuracy of the F1 score for every k-length combination of
                    authors which are specified by input. For example:
                    authorial_decomposition(['friedman', 'krugman', 'dowd'],
                                             k=2) = [0.979, 0.915, 0.988]

                    Here, 
                        authorial_decomposition(['friedman', 'krugman']) = 0.979,
                        authorial_decomposition(['friedman', 'dowd']) = 0.915,
                        authorial_decomposition(['krugman', 'dowd']) = 0.988.

    Prints:
        Final accuracy value and incremental infomation on function progress.
    """
    if k != None:
        perms = itertools.combinations(authors, k)
        return [authorial_decomposition(i, POS, None, randomize, num_runs,
                                        ngram_range) for i in perms]
    else:
        n = len(authors)
        total = 0
        matrix, true_labels = get_matrix(authors, POS,
                                         ngram_range), get_labels(authors, POS)
        for i in range(num_runs):
            if randomize:
                matrix, true_labels, order = randomize_data(
                    matrix, np.array(true_labels))
            predicted_labels = cluster(n, POS, matrix)
            total += cluster_accuracy(predicted_labels, true_labels, n)
            sys.stdout.write('\rCompleted Prediction Run: %d' % (i + 1))
            sys.stdout.flush()
        string = '\n' + '{0}: {1}'.format(
            str(authors), total / float(num_runs)) + '\n'
        sys.stdout.write('\r%s' % string)
        sys.stdout.flush()
        return total / float(num_runs)


def _parse_command_line_input(cl_input):
    authors = cl_input[1].replace('--authors=', '').split(',')
    authors = [a.strip() for a in authors]
    pos, k, randomize, num_runs, ngram_range = True, None, True, 10, (3, 5)
    for i in range(2, len(cl_input)):
        if '--pos=' in cl_input[i]:
            pos = cl_input[i].replace('--pos=', '').strip() == 'True'
        if '--k=' in cl_input[i]:
            k = int(cl_input[i].replace('--k=', '').strip())
        if '--randomize=' in cl_input[i]:
            randomize = cl_input[i].replace('--randomize=',
                                            '').strip() == 'True'
        if '--num_runs=' in cl_input[i]:
            num_runs = int(cl_input[i].replace('--num_runs=', '').strip())
        if '--ngram_range' in cl_input[i]:
            ngram_range = eval(cl_input[i].replace('--ngram_range=', ''))
    return authors, pos, k, randomize, num_runs, ngram_range


if __name__ == '__main__':
    authors, pos, k, randomize, num_runs, ngram_range = _parse_command_line_input(
        sys.argv)
    authorial_decomposition(authors, pos, k, randomize, num_runs, ngram_range)
