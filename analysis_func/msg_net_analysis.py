import pandas as pd
import numpy as np
from networkx.algorithms.community.modularity_max import greedy_modularity_communities as gmc
import networkx as nx
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

# function to identify influential nodes
def influ_nodes(G,df_G, n=10,start_date = datetime(2006,1,1), end_date = datetime.now()):
    # create subgraph of links within specified time frame
    sub_G = nx.DiGraph()
    sub_G.add_nodes_from(G.nodes.items())
    subsect_edges = [(i[0],i[1],attrdict) for i,attrdict in G.edges.items() if
     pd.Timestamp.to_pydatetime(attrdict['date'])>= start_date and
     pd.Timestamp.to_pydatetime(attrdict['date'])<=end_date]
    sub_G.add_edges_from(subsect_edges)
    # remove isolated nodes from subgraph
    isolates = [i for i in nx.isolates(sub_G)]
    sub_G.remove_nodes_from(isolates)

    # start a results dict
    results={}
    # generate dataframes of influential nodes based on different criteria
    # most prolific tweeters
    outdeg_df = pd.DataFrame([[node,val] for (node, val) in sorted(sub_G.out_degree, key=lambda x: x[1], reverse=True)])
    outdeg_df.columns = ['Account', 'Tweets sent']
    results['tweeters'] = outdeg_df.head(n)

    # most prolific receivers of tweets
    indeg_df = pd.DataFrame([[node,val] for (node, val) in sorted(sub_G.in_degree, key=lambda x: x[1], reverse=True)])
    indeg_df.columns = ['Account', 'Tweets received']
    results['tweet_receive'] = indeg_df.head(n)

    # participants in conversation with largest following
    sub_G2 = sub_G.subgraph([i for i,j in sub_G.out_degree if j > 0])
    follow_df = pd.Series(nx.get_node_attributes(sub_G2, 'follower_count'))
    follow_df = follow_df.sort_values(ascending=False)
    results['followers'] = follow_df.head(n)

    # participants who generated the most interactions
    # use dataframe of tweets for this
    act_df = df_G[['author_name','retweet_count','favorite_count']].copy()
    act_df['interact_count'] = act_df.retweet_count + act_df.favorite_count
    group_df = act_df.groupby('author_name').sum().sort_values('interact_count',ascending=False)
    results['gen_interact'] = group_df.head(n)
    return(results)

def find_communities(G,start_date = datetime(2006,1,1), end_date = datetime.now()):
    # create subgraph of links within specified time frame
    sub_G = nx.DiGraph()
    sub_G.add_nodes_from(G.nodes.items())
    subsect_edges = [(i[0],i[1],attrdict) for i,attrdict in G.edges.items() if
     pd.Timestamp.to_pydatetime(attrdict['date'])>= start_date and
     pd.Timestamp.to_pydatetime(attrdict['date'])<=end_date]
    sub_G.add_edges_from(subsect_edges)
    # remove isolated nodes from subgraph
    isolates = [i for i in nx.isolates(sub_G)]
    sub_G.remove_nodes_from(isolates)
    communities = gmc(sub_G.to_undirected())

    return(communities)

# function to calculate reach of a messaging network
def get_reach(G, start_date = datetime(2006,1,1), end_date = datetime.now()):
    # create subgraph of links within specified time frame
    sub_G = nx.DiGraph()
    sub_G.add_nodes_from(G.nodes.items())
    subsect_edges = [(i[0],i[1],attrdict) for i,attrdict in G.edges.items() if
         pd.Timestamp.to_pydatetime(attrdict['date'])>= start_date and
         pd.Timestamp.to_pydatetime(attrdict['date'])<=end_date]
    sub_G.add_edges_from(subsect_edges)
    # remove isolated nodes from subgraph
    isolates = [i for i in nx.isolates(sub_G)]
    sub_G.remove_nodes_from(isolates)

    # impressions - number of people this appeared in their time line for
    # followers of original authors + followers of retweeters
    # this is sum of followers amongst nodes with out degree > 1
    sub_G2 = sub_G.subgraph([i for i,j in sub_G.out_degree if j > 0])
    follower_nums = [attr['follower_count'] for i,attr in sub_G2.nodes.items()]
    impressions = np.nansum(follower_nums, dtype = 'int')

    # interactions - number of people that directly interacted with tweets
    # retweeting or favoriting are the only interactions we have a direct view of
    # so simple sum of those amongst edges
    interactions = np.nansum([attr['favorites'] for i,attr in sub_G.edges.items()] +
        [attr['retweets'] for i,attr in sub_G.edges.items()], dtype = 'int')

    return({'impressions':impressions, 'interactions':interactions})


# function to create a graph of reach over time
def reach_graph(G, start_date, end_date, time_step = 'days'):
    # set time step
    if time_step == 'days':
        step = timedelta(days=1)
    if time_step == 'weeks':
        step = timedelta(weeks=1)
    if time_step == 'hours':
        step = timedelta(hours=1)
    _date = start_date
    # iterate through reach for each time step and store reach results in dataframe
    df = pd.DataFrame(columns = ['date','impressions','interactions'])
    while _date < end_date:
        _date += step
        result = get_reach(G, start_date = start_date, end_date= _date)
        result['date'] = _date
        df = df.append(pd.Series(result), ignore_index = True, sort= False)
    # columns for marginal change
    df['impressions_chg'] = df['impressions'].diff()
    df['interactions_chg'] = df['interactions'].diff()

    # columns for percentile of maximum impressions/interactions
    df['impressions_ptile'] = df['impressions']/df['impressions'].max()
    df['interactions_ptile'] = df['interactions']/df['interactions'].max()

    #TODO: turn df into a matplotlib graph
    return(df)
