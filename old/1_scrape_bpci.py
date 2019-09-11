# extract all FDA announcements
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.insert(0, abspath(join(dirname(__file__), '..')) + '\\extract_func')

from extract_tweets import extract_tweets
import pandas as pd

tweet_df = extract_tweets('BPCI',['BPCI'],'2018-05-15',
    None, get_replies = False, get_retweets = False)

tweet_df.to_csv("output/BPCI_tweets.csv",index=False)
