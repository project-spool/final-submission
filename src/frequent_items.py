# frequent-items.py
# author: Tyler Angert
# given a sample user cluster, finds frequent artists among different clusters

from pymining import itemmining
import math

def find_frequent_artists(sample_clusters):

    """
     Finds frequent artists from a sample cluster object of users, cluster labels, and artist data
    """

    print("Finding frequent item sets")
    print(sample_clusters)

    # sample cluster data on 5000 random american users, k = 10 for k means, and top 5 artists
    frequent_artist_dict = {}

    for cluster, user_data in sample_clusters:

        print("\nFinding frequent artists for cluster {}".format(cluster))

        num_users = len(user_data.user_id)

        # calculates the minimum support of artists according to some proportion of users
        # ex: pass in 10, so min support is num users / 10, or 10% of users
        # for some reason we can't import this number as a parameter...?
        min_sup = math.floor(num_users/5)

        if min_sup == 0:
            min_sup = 1

        # this is for humongous clusters where a large minimum support ( > 300 ) doesn't really make sense
        # for the Last.fm data set
        # if num_users > 1000:
        #     min_sup = num_users/20

        # print("min sup: ", min_sup)
        # print("number of users: {}".format(num_users))

        # create a list of "transactions" for frequent mining from the top artists for the current user
        transactions = (list(user_data.top_artists))
        relim_input = itemmining.get_relim_input(transactions)

        # the report stores each frequent item as a dictionary of the form:
        # frozenset(artist id, artist name) : count
        report = itemmining.relim(relim_input, min_support=min_sup)

        # each frequent item is stored as a frozen set
        # process each frozen set item by converting it into a list and accessing the data
        # (through the 0th index, because it's a list with just 1 element)
        # then grabbing just the artist name through the 1st index
        # (because it is the 2nd item in the (artist ID, artist name) tuple for each frozen set

        report = [(list(item)[0][1], report[item]) for item in report if len(item) == 1]

        # sort the report object in reverse order so the highest played artists are first
        report = sorted(report, key=lambda tup: tup[1], reverse=True)
        # print(report)

        # store the report list for the cluster number in the frequent artist dictionary
        frequent_artist_dict[cluster] = report

    return frequent_artist_dict