# process.py
# author: Tyler Angert
# pre-processes and cleans input data to get ready for clustering

import math
from collections import Counter
import random
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from globals import ALL_USER_PROFILES, AMERICAN_USERS


################################
# MARK: Pre-processing functions
################################

def filter_incompletes(df):
    """
     Drops all empty records from a data frame
    """
    return df.dropna()


def sample(size, df):
    """
     Samples a data frame
    """
    return df.sample(n=size)


def convert_to_df(path):

    """
     Converts a tsv file to a data frame
    """
    return pd.read_csv(path, sep='\t')


def create_random_user_tsv(num, users):

    """
     Creates a tsv file of random users, given a sample size and user data frame
    """

    # user_ids = user_artist_df['user_id'].unique()
    # create_random_user_tsv(10000, user_ids)

    random_users = random.sample(list(users), num)
    random_users_df = pd.DataFrame(random_users, columns=['user_id'])
    random_users_df.to_csv('../../data/random_users.tsv', sep='\t', index=False)


def tf_idf_dict():
    return 0

# FIXME: Not complete
def tf_idf(artist, parent_document, all_documents):

    """
     Calculates the tf-idf score for each artist
    """

    # TF(artist) = (Number of times user plays an artist) / (Total number of plays for user).
    tf = 5

    # IDF(artist) = log_e(Total number of users / Number of users who play an artist).

    idf = math.log(5, math.e)

    return tf * idf


def grab_country_users(country, user_profile_df):

    """
     Returns a data frame of country specific users given a complete user profile dataframe.
    """

    return user_profile_df[user_profile_df.country == country]


def grab_all_user_profiles(path):

    """
     Returns a incompelte-filtered data frame for all user profiles
    """

    user_profs_df = pd.read_csv(path, sep='\t')
    return filter_incompletes(user_profs_df)


def get_artist_metadata(user_ids, user_id_groups, top_artist_count):

    """
     Maps artist ids to artist names
    """
    #maps artist id's to names
    artist_id_dict = {}

    # stores artist ID's and how many users have them
    artist_user_counter = Counter()

    print("Creating artist-id dictionary")
    print("NOTE: For some reason, some of the users get lost in the creation of this dictionary, but the effect is negligible on large data sets (< 0.5% of users)")

    for uid in user_ids:
        try:
            group = user_id_groups.get_group(uid)
        except KeyError as err:
            print("Couldn't find user")

        top_data = group.head(top_artist_count)
        artist_ids = list(top_data['artist_id'])
        artist_names = list(top_data['artist_name'])
        zipped = zip(artist_ids, artist_names)

        for id in artist_ids:
            artist_user_counter.update([id])

        for k, v in zipped:
            artist_id_dict[k] = v

    return artist_user_counter, artist_id_dict


# MARK: Main process function that returns the appropriate USER x ARTIST matrix (explained inside)
def process(sample_size, top_artist_count):

    print("Pre processing data")

    """
    Big picture plan:

    The raw data from Last.fm is formatted in the following way:

    1. User profile data
    Contains ~360,000 unique users of the form:
    [ UserID, Gender, Age, Country, Join Date]

    2. User-artist data
    Contains ~17,000,000 lines of each user and their top artists (approximately 50 each)
    [ UserID, ArtistID, Artist Name, Play Count]

    In order to cluster the users by top artist, we essentially will create a
    large USER x ARTIST matrix that looks like this:

                    ARTIST

                    artist_1     artist_2     artist_3     ...     artist_n

    USER    user_1      1           0           0                      1

            user_2      0           1           1                      0

            user_3      1           1           0                      1

            ...

            user_n      1           0           1                      0


    Where each user is represented by an n-dimensional vector of artists they
    listen to (represented by either a 1 (boolean value) or the ratio of how often that user listens
    to an artist.


    Then, using various clustering algorithms (for now, KMeans and Euclidean distance), cluster all of the users
    based on their vector distances.

    Once the users are clustered, we will use frequent itemset mining to capture the top artists within each cluster,
    and classify new users into the clusters (listening groups, essentially) using KNN.

    Current plan is to cluster on American users, and then look at other countries.

    """

    """
    ###################################################
    # MARK: Create and sample initial data sets
    # Only used at the beginning to get all of the data
    ###################################################
    """

    # # Grab all users
    # all_user_profs_df = grab_all_user_profiles('../data/raw/usersha1-profile.tsv')
    # all_user_profs_df.columns = ['user_id', 'gender', 'age', 'country', 'join_date']
    #
    # # Grab american users
    # american_user_df = grab_country_users('United States', all_user_profs_df)
    #
    # # Export american users as tsv
    # all_user_profs_df.to_csv('../data/cleaned/united-states-users.tsv',sep='\t')
    #
    # AMERICAN_USERS = '../data/cleaned/united-states-users.tsv'
    # american_users_df = convert_to_df(AMERICAN_USERS)
    # american_users_df.to_pickle('../data/pickles/american-users-pickle.pkl')

    """
    ###############################################################################
    # MARK: Grab pre processed data from pickles (stored/serialized python objects)
    ###############################################################################
    """

    american_users_df = pd.read_pickle('../data/pickles/american-users-pickle.pkl')
    random_americans = sample(sample_size, american_users_df)

    user_ids = random_americans.user_id.unique()
    user_id_groups = pd.read_pickle('../data/pickles/user-id-groups.pkl')

    TOP_ARTIST_COUNT = top_artist_count

    """
    ###############################################################
    # MARK: Setup reference dictionaries for artist IDs and indices
    ###############################################################
    """

    # Used to reference back IDs and artist names
    artist_user_counter, artist_id_dict = get_artist_metadata(user_ids, user_id_groups, TOP_ARTIST_COUNT)

    # Used to reference artist IDs to their spots in the user x artist matrix
    artist_index_dict = {}

    # Stores artists and their indices in the user artist matrix
    # acts as a two-way dictionary
    for idx, artist in enumerate(artist_id_dict.keys()):
        artist_index_dict[artist] = idx
        artist_index_dict[idx] = artist

    # column names for the pandas artist data frame
    artist_df_cols = ['artist_id', 'artist_name', 'play_count']

    """
    ###############################################################
    ###############################################################

    # MARK: Setup row and column creation for user x artist matrix
        Using a CSR (Compressed Sparse Row matrix)
        This stores a sparse matrix (mostly 0's) in the form of three 1D arrays:

        Array 1: row indices
        Array 2: column indices
        Array 3: data

        So when you read down all three arrays vertically, you essentially get "coordinates" for each data point
        inside of the fully populated (dense) matrix.

        Although our initial model mathematically represents our user vectors as columns, we will represent
        our users as rows so we can easily store the matrix as a 2D array and pass it into scikit-learn's packages.

    ###############################################################
    ###############################################################
    """

    # Populate each of these numpy arrays as you loop through each of the users and their data

    # rows = coordinates for the USERS in the user x artist matrix
    row_indices = np.array([])

    # columns = coordinates for the ARTISTS in the user x artist matrix
    col_indices = np.array([])

    # stores the vector weights at each USER x ARTIST coordinate
    user_artist_data = np.array([])

    #################################
    #################################
    #################################

    # In order to populate each of the arrays, we keep tracker variables
    # for the current row, current column, and current data as we iterate through each user and artist

    # keeps track of the current user
    current_row = 0

    # keeps track of the current artist
    current_col = 0

    # keeps track of the current listening weight
    current_data = 0

    print("Users analyzing: {}".format(len(user_ids)))

    # For some reason, some of the UserIDs are null in the larger dataframe, so we keep track of the
    # found users and their data in a new dictionary
    found_users = {}

    # keeps track of the null/not found users
    err_count = 0

    """
    ###############################################################
    # MARK: Go through all of the user ids and grab relevant data
    for each user-artist combo into the sparse matrix
    ###############################################################
    """
    # stores the IDF values for each
    IDF_dict = {}

    # IDF(artist) = log_e(Total number of users / Number of users who play an artist).
    NUM_USERS = len(user_ids)

    """
     need to iterate through artists
     for each artist:
        count how many users have that artist

    """
    # iterate through each of the relevant users
    # increment "current_row" on each now uid loop in order to keep track of the coordinates
    for uid in user_ids:

        # try to grab the croup, otherwise skip over this iteration of the loop
        try:
            # grab the group
            group = user_id_groups.get_group(uid)

            # initialize the founder users dict with an array to store data
            found_users[uid] = []

        except KeyError:
            err_count += 1
            print("couldn't find {} users".format(err_count))
            continue

        # grabs the top 5, 10, 15 pandas rows from the top artist count
        top_data = group.head(TOP_ARTIST_COUNT)

        # grabs the relevant subset of data from the specified artist datafram columns
        # [artist id, artist name, play count]
        artist_data = top_data[artist_df_cols]

        # FIXME: include tf-idf to discount large globally popular artists (like the beatles)
        # grab total plays to calculate the normalized playcount
        total_plays = top_data['play_count'].sum()

        # stores all of the tuples
        artist_data_tuples = list()

        # iterate through all of the rows of form: [artist id, artist name, play count]
        for d in artist_data.values:
            # takes in the artist id and gets the appropriate index
            artist_id = d[0]
            artist_index = artist_index_dict[artist_id]
            play_count = d[2]

            # set the current column to the artist index from the index dictionary
            current_col = artist_index

            # the current "term frequency"
            # this is the TF term
            tf = (play_count/total_plays)

            # get the total number of plays
            total_user_plays = artist_user_counter[artist_id]
            idf = math.log(NUM_USERS/total_user_plays)

            # this discounts globally popular artists from being in every cluster
            # like the beatles and radiohead
            # currently makes the clusters worse for some reason...so just use tf instead
            current_data = (tf * idf)

            # now it's time to add the new data to the data arrays
            row_indices = np.append(row_indices, current_row)
            col_indices = np.append(col_indices, current_col)
            user_artist_data = np.append(user_artist_data, tf)

            # store all of the relevant data into a tuple
            tup = (artist_id, artist_id_dict[artist_id])
            artist_data_tuples.append(tup)

        # store the list of top artist tuples into each found user
        found_users[uid] = artist_data_tuples

        # increment current user row
        current_row += 1

    """
    ###############################################################
    # MARK: Once all three 1D arrays are populated, pass them into
    scikit-learns csr_matrix function and create the matrix
    ###############################################################
    """

    # create the row-centric matrix
    user_artist_mtx = csr_matrix((user_artist_data, (row_indices, col_indices)))

    #

    # return the found users dictionary and dense user artist matrix to pass into clustering
    return found_users, user_artist_mtx.toarray()