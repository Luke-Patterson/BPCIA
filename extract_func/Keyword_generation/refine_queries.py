import pandas as pd
import numpy as np

df = pd.read_pickle('output/search_phrases.pkl')
ann_df = pd.read_excel('FDA_Announcements.xlsx')

phrase_df = pd.DataFrame(index=df.index, columns=range(3))
for i,j in df.iterrows():
    row = j.dropna().reset_index(drop=True)
    phrase_df.iloc[i,:] = row
