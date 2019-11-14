# define a user object
class User(object):
    def __init__(self, screen_name, follower_count):
        self.screen_name = screen_name
        self.follower_count = follower_count

# define tweet objects
class Tweet(object):
    def __init__(self, user_mentions, rt_names,author_name):
        self.user_mentions = user_mentions
        self.rt_names = rt_names
        self.author_name = author_name


# function to build a messaging network from a set of tweets
def build_msg_net(tweet_df):
    import pandas as pd
    import numpy as np
    import json
    import networkx as nx
    from ast import literal_eval
    def _leval(x):
        try:
            return literal_eval(x)
        except:
            return None

    # transform dict/list strings to actual dict/list objects
    tweet_df['user_mentions'] = tweet_df['user_mentions'].apply(_leval)
    tweet_df['rt_names'] = tweet_df['rt_names'].apply(_leval)
    tweet_df['rt_follower_counts'] = tweet_df['rt_follower_counts'].apply(_leval)

    # look for the "date" variable in the data set
    if 'created_at' in tweet_df.columns:
        date_var='created_at'
    elif 'date' in tweet_df.columns:
        date_var='date'
    else:
        raise('no date var name found in tweet df. Expected either "date" or "created_at"')
    # transform created_by to a datetime value
    tweet_df[date_var] = pd.to_datetime(tweet_df[date_var])

    # get unique list of tagged and author accounts
    # authors - just take unique values of data frame
    authors = tweet_df[['author_name', 'author_followers']]
    authors = authors.drop_duplicates('author_name')
    # create a dictionary of user objects to store user information
    user_dict = {}
    for i,row in authors.iterrows():
        user_dict[row.author_name] = User(row.author_name,row.author_followers)

    # mentions - need to manipulate some because there are multiple mentions per tweet sometimes
    mentions = tweet_df['user_mentions'].values.tolist()
    if mentions is not None and not all([i is None for i in mentions]) :
        # flatten list
        mentions = [i for j in mentions for i in j if j]

        # if not already in users list, add to it. Don't know number of followers for mentions
        # Followers are not counted as impressions anyways
        for i in mentions:
            if i not in user_dict.keys():
                user_dict[i]= User(i,follower_count=np.nan)

    # do the same for retweeters retweeting users, but we also need to add their
    # followers, since retweets do create impressions
    retweeters = tweet_df[['rt_names','rt_follower_counts']]
    for i,row in retweeters.iterrows():
        if row.rt_names is not None:
            for j,k in zip(row.rt_names, row.rt_follower_counts):
                if j not in user_dict.keys():
                    user_dict[j]= User(j,int(k))


    # create networkx network
    G= nx.DiGraph()
    # add users as nodes
    G.add_nodes_from([(node, {'follower_count': attr.follower_count}) for
        (node, attr) in user_dict.items()])
    # draw edges between users mentioning other users
    for i, row in tweet_df.iterrows():
        if mentions is not None and not all([i is None for i in mentions]):
            for j in row['user_mentions']:
                G.add_edge(row.author_name,j,
                    favorites = row['favorite_count'],
                    retweets = row['retweet_count'],
                    date = row[date_var])
        # draw edges between users retweeting them
        if row['rt_names'] is not None:
            for j in row['rt_names']:
                G.add_edge(row.author_name,j, date = row[date_var],
                retweets = 0,
                favorites = 0)
        # if no users are mentioned, create a self edge for the tweet
        if row['user_mentions'] == []:
            G.add_edge(row.author_name, row.author_name,
                favorites = row['favorite_count'],
                retweets = row['retweet_count'],
                date = row[date_var])
    return(G)
