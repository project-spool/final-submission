# pipeline.py
# author: Tyler Angert
# hosts all processing and clustering processes

from process import process
from cluster import optimize, calc_sses, plot_sse, create_cluster_groups, get_all_results
from frequent_items import find_frequent_artists
import pandas as pd
from cluster import NoIndentEncoder
import json

def run_processing_pipeline(sample_size, top_artist_count):
    """
     Runs the pre processing algorithms to produce the user artist matrix
    """
    return process(sample_size=sample_size, top_artist_count=top_artist_count)


def run_clustering_pipeline(user_artist_mtx, START, STOP):

    """
     Runs MiniBatch K means given a sample size and top artist count
    """
    # Pass both into the cluster function
    return calc_sses(user_artist_mtx, START, STOP)


def run_frequent_items_pipeline(cluster_groups):

    """
     Finds the frequent item sets from a given pickle
    """
    return find_frequent_artists(sample_clusters=cluster_groups)

def data2json(frequent_artists, size):
    print("Converting data to JSON for visualization")

    # Initial setup
    viz_dict = {}
    viz_dict["name"] = '{} Last.fm Users'.format(size)
    viz_dict["children"] = []

    # Go through each of the children
    for k in frequent_artists:

        current_cluster = {}
        artists = frequent_artists[k]

        children = []

        for artist in artists:
            artist_obj = {}
            artist_obj["name"] = artist[0]
            artist_obj["size"] = artist[1]
            children.append(artist_obj)

        # gets the name of the top artist
        if len(artists) == 0:
            name = ""
        else:
            name = artists[0][0]

        current_cluster["name"] = name
        current_cluster["children"] = children

        viz_dict["children"].append(current_cluster)

    return viz_dict


def export_json(dict, filename):
    with open(filename, 'w') as f:
        output = json.dump(dict, f, indent=4, ensure_ascii=False, separators=(',', ': '))
        f.write('\n')

def test():

    #################
    # PARAMETER SETUP
    #################

    # How many top artists you grab from each user
    TOP_ARTIST_COUNT = 15

    # The sample size form the ~40k or so American users
    SIZE = 5000

    # Minimum number of clusters
    START = 5

    # Maximum number of clusters
    STOP = 15

    ##################
    # RUN THE PIPELINE
    ##################

    # Error catching
    if SIZE > 40000:
        return "Sorry, choose a smaller sample size to cluster"

    if STOP == 0:
        return "Sorry, choose a higher number of initial clusters"

    if TOP_ARTIST_COUNT > 20:
        return "Sorry, please choose a lower number of top artists"

    # Pipeline creation
    found_users, user_artist_mtx = run_processing_pipeline(SIZE, TOP_ARTIST_COUNT)

    # Grab the results from the clustering
    sse = run_clustering_pipeline(user_artist_mtx, START, STOP)

    # Format the results from the sse object
    results = get_all_results(sse)

    # EXPORT THE CLUSTERING RESULTS
    export_json(results, 'results/clustered-user-results-{}-americans.json'.format(SIZE))

    # iterate through each of the cluster objects,
    # grab the frequent artists,
    # and export the visualization data for d3
    for result in results:

        cluster_groups = create_cluster_groups(found_users, result["labels"])

        frequent_artists = run_frequent_items_pipeline(cluster_groups)

        viz = data2json(frequent_artists, SIZE)

        # EXPORT THE VISUALIZATION TREES
        export_json(viz, 'results/clustered-users-{}-{}-americans.json'.format(result['k'], SIZE))

if __name__ == '__main__':
    test()