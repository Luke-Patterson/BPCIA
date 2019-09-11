# functions to analyze sentiment of tweets
import pandas as pd
import numpy as np
from nltk.corpus import opinion_lexicon as ol
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation

# add sentiment scores from Vader to a tweet data frame
def sentiment_scoring(df, text_var='full_text'):
    sentim_analyzer = SentimentIntensityAnalyzer()
    df = pd.concat([df, pd.DataFrame(list(df[text_var].apply(lambda x:
        sentim_analyzer.polarity_scores(x))))],axis=1)
    return(df)
