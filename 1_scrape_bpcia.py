# Request for Twitter scraping from apprenticeship project 10/2

from extract_func.extract_tweets import extract_tweets
import pandas as pd
import numpy as np

df=extract_tweets('BPCI',['BPCIAdvanced','"BPCI Advanced"','"BPCIA"',
    '"BPCI-A"','"Bundled Payments for Care Improvement–Advanced"',
    '"Bundled Payments for Care Improvement Advanced"'],
   get_replies = False, get_retweets = False, geocode_loc=True)
df.to_csv('output/BPCIA_tweets_raw.csv')
