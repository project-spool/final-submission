# cluster.py
# author: Tyler Angert
# takes a user x artist matrix and performs k-means clustering

import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import numpy as np

import re
import json
import _ctypes
import uuid

"""
Used for JSON exports
"""

class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in self._replacement_map.iteritems():
            result = result.replace('"@@%s@@"' % (k,), v)
        return result

def cosine(X, Y=None, Y_norm_squared=None, squared=False):
    return cosine_distances(X,Y)


def calc_sses(matrix, start, stop):
    """
     Calculates the SSEs for an input matrix between start and stop k values
    """
    sse = {}

    for k in range(start, stop+1):

        print("\nRunning on k = {}".format(k))
        # assign a cosine distance function
        k_means_.euclidean_distances = cosine

        # call the clustering
        km = MiniBatchKMeans(n_clusters=k)
        km = km.fit(matrix)

        print("Inertia: {}".format(km.inertia_))

        sse[k] = (km.inertia_, km.labels_.tolist())

    return sse


def get_all_results(sse):

    results = []
    kv_pairs = zip(sse.keys(), sse.values())

    for pair in kv_pairs:

        result_dict = {
        "k": pair[0],
        "sse": pair[1][0],
        "labels": pair[1][1]
        }

        results.append(result_dict)

    return results



def optimize(sse):

    """
     Takes in a dictionary of all SSEs for given k's and returns a tuple of the min SSE and their cluster labels
    """
    kv_pairs = zip(sse.keys(), sse.values())
    min_result = min(kv_pairs, key=lambda pair: pair[1])

    result_dict = {
        "k": min_result[0],
        "sse": min_result[1][0],
        "labels": min_result[1][1]
    }

    return result_dict


def plot_sse(sse):

    plt.figure()
    plt.interactive(False)
    plt.plot(list(sse.keys()), [val[0] for val in sse.values()])
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.title("K-Means SSE for 20,000 Last.fm users")
    plt.show()


def create_cluster_groups(found_users, cluster_labels):

    # stores all of the user-cluster label data to be passed into a new pandas data frame
    all_tuples = []

    # iterates through the user-cluster tuple combinations, then references the found users to get top artists
    for user, clust in zip(found_users.keys(), cluster_labels):
        top_artists = found_users[user]
        tup = (clust, user, top_artists)
        all_tuples.append(tup)

    # creates a data frame from all the tuples
    user_cluster_df = pd.DataFrame(all_tuples)
    user_cluster_df.columns = ['cluster', 'user_id', 'top_artists']

    print("cluster value counts: ")
    # print(user_cluster_df.cluster.value_counts())

    # groups all users by cluster for easy analysis
    cluster_groups = user_cluster_df.groupby('cluster')

    # for name, group in cluster_groups:
    #     print(group)
    #     print(name)
    #     print(group)
    #     print('\n')

    # pass the grouped clusters to the frequent itemset mining
    return cluster_groups