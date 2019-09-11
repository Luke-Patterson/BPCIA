import os, sys
import pandas as pd
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# get user objects for retweeters for test data set
import pandas as pd
from got_supplement_API_data import get_API_data
from got_supplement_API_data import get_retweeters_names

# load current test data set
df = pd.read_csv('test_output/got_test_output_full_rev.csv')

df = get_API_data(df['id'].tolist(), get_retweets = True)

df.to_csv('test_output/got_test_output_full_rev_v2.csv', index = False)
