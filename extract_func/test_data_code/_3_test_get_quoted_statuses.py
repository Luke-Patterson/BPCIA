# fixing up test data file
import os, sys
import pandas as pd
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from reply_scraping import reply_scrap
from got_supplement_API_data import get_API_data
import numpy as np
from ast import literal_eval
from clean_API_output import clean_API_output
# take quoted statuses, and append

tweet_df = pd.read_csv('test_output/got_test_output_full.csv')

# take string dicts from quoted status column, deduplicate, and turn them into dictionary objects
def _leval(x):
    try:
        return literal_eval(x)
    except:
        return None

clean_stats = tweet_df.quoted_status.dropna().unique()
clean_stats = [_leval(i) for i in clean_stats]

# Then, create a dataframe from the list of dictionaries
for i, chunk in enumerate(clean_stats):
    if i == 0:
        stats_df = pd.DataFrame(pd.Series(chunk)).transpose()
    else:
        stats_df = stats_df.append(pd.DataFrame(pd.Series(chunk)).transpose(), ignore_index=True, sort= False)

# finally, add to the original dataframe of tweets
tweet_df = tweet_df.append(stats_df, ignore_index = True, sort = False)
tweet_df = tweet_df.drop_duplicates('id')

# create permalinks
tweet_df['permalink'] = 'https://twitter.com/statuses/' + tweet_df['id_str'].astype('str')

# # get replies
# print('getting replies')
# tweet_df = reply_scrap(tweet_df)
#
# print('getting "in reply to" tweets')
#
# # get tweets that are found as in reply to in the original set of tweets
# add_tweets = tweet_df.in_reply_to_status_id_str.unique().tolist()
# if add_tweets != [None]:
#     if None in add_tweets:
#         add_tweets.remove(None)
#     add_tweets_df = get_API_data(add_tweets, get_retweets = False)
#     tweet_df = tweet_df.append(add_tweets_df, ignore_index=True, sort=False)

tweet_df = clean_API_output(tweet_df)
tweet_df.to_csv('test_input/got_test_output_full_rev.csv', index = False)
