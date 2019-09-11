from extract_tweets import extract_tweets
from extract_tweets import extract_counts
from extract_tweets_api import extract_tweets_api
from extract_tweets_api import extract_counts_api
from got_supplement_API_data import get_API_data
from reply_scraping import reply_scrap
import pandas as pd
# Standard API extraction
# single page
# api_df = extract_tweets_api('Mammography',
#     ['fda%20mammography'],
#     "2019-04-01")
# api_df.to_csv('test_output/api_test_output.csv', index = False, encoding='utf-8')

# multi-page
# api_df = extract_tweets_api('Medtronic',
#     ['fda%20Medtronic'],
#     "2019-04-05")
# api_df.to_csv('test_output/api_test_output_multi.csv', index = False, encoding='utf-8')

# 30 day search
# api_df = extract_tweets_api(title = 'Medtronic',
#     keywords = ['fda Medtronic'],
#     start_date="201904250000",
#     search = '30day')
# api_df.to_csv('test_output/api_test_output_30day.csv', index = False, encoding='utf-8')

# full archive
# api_df = extract_tweets_api(title = 'Duodenoscope',
#     keywords = ['fda Duodenoscope Reprocessing Instructions'],
#     search = 'fullarchive')
# api_df.to_csv('test_output/api_test_output_full_archive.csv', index = False, encoding='utf-8')

# counts of tweets
# api_df = extract_counts_api(title = 'Duodenoscope',
#      keywords = ['fda Duodenoscope Reprocessing Instructions'],
#      search = 'fullarchive')
# result[df]
# api_df.to_csv('test_output/api_test_output_counts.csv', index = False, encoding='utf-8', header=True)

# GoT Extraction
# test_df = extract_tweets(title = 'test',
#      keywords = ['FDA Juul'],
#      start_date = "2019-02-01",
#      get_replies = True,
#      get_retweets = False)
# test_df.to_csv('test_output/got_test_output_full.csv', index = False, encoding='utf-8')

# test_df = extract_tweets(title = 'test',
#      keywords = ['FDA Mammography'],
#      start_date = "2019-04-01",
#      get_replies = True,
#      get_retweets = False)
# test_df.to_csv('test_output/got_test_output.csv', index = False, encoding='utf-8')

# GOT tweet counts
# test_counts = extract_counts('Health Affairs',
#     ['FDA%20Juul'],
#     start_date = "2019-01-01",
#     get_replies = True)
# print(test_counts)

# note replies with selenium scraping
#test_df2 = reply_scrap(pd.read_csv('test_output/got_test_output.csv'))

# getting retweets from API
# full_df = get_API_data(test_df['id'].values)
# full_df.to_csv('test_output/GoT_tweets_API_suppl_output.csv', encoding='utf-8', index=False)
