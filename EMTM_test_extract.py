from extract_func.extract_tweets import extract_tweets
import pandas as pd

df=extract_tweets('EMTM',['"Enhanced MTM"','"Enhanced Medication Therapy Management"'],
'2015-01-01', None, get_replies = False, get_retweets = False, geocode_loc=False)
df.to_csv("output/EMTM_tweets.csv",index=False)
