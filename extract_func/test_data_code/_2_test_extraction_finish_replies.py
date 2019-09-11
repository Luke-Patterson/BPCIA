# fixing up test data file
import os, sys
import pandas as pd
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# finish scraping
from reply_scraping import reply_scrap
from got_supplement_API_data import get_API_data
import pandas as pd

test_df = pd.read_csv('test_output/got_test_output.csv', encoding='utf-8')

finished_df = test_df.loc[(test_df['replying_users'].isna() == False)]

unfin_df = test_df.loc[(test_df['replying_users'].isna())]

unfin_df = reply_scrap(unfin_df)

tweet_df = finished_df.append(unfin_df, ignore_index=True, sort=False)
add_tweets = tweet_df.in_reply_to_status_id_str.unique().tolist()
if add_tweets != [None]:
    if None in add_tweets:
        add_tweets.remove(None)
    add_tweets_df = get_API_data(add_tweets, get_retweets = True)
    tweet_df = tweet_df.append(add_tweets_df, ignore_index=True, sort=False)
tweet_df.to_csv('test_output/got_test_output_full.csv', index = False, encoding='utf-8')
