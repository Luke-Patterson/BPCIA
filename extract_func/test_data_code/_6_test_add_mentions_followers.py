# clean up final test data set
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# get user objects for retweeters for test data set
import pandas as pd
from clean_API_output import clean_API_output

df = pd.read_csv('test_output/got_test_output_full_rev_v2.csv')

df.to_csv('test_output/got_test_output_full_rev_v3.csv',index=False)
