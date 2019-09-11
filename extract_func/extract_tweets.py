# extract tweets with Get Old Tweets library

def extract_tweets(title, keywords, start_date=None, end_date=None,
    get_replies = True, get_retweets = False, halt_count=None, geocode_loc=False):
    from . import got3_selen as got
    import datetime
    import pandas as pd
    from .reply_scraping import reply_scrap
    from .got_supplement_API_data import get_API_data
    from .clean_API_output import clean_API_output
    print("extracting " + title + " tweets")
    start = datetime.datetime.now()
    for count, word in enumerate(keywords):
        print("searching keyword: " + word)
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(word)
        if pd.isna(start_date) == False:
            if start_date[4] != '-' or start_date[7] != '-' or len(start_date) != 10:
                raise('Error: start_date not in correct format (YYYY-MM-DD),' +
                    'start_date is' + str(start_date))
            tweetCriteria = tweetCriteria.setSince(start_date)
        if pd.isna(end_date) == False:
            if end_date[4] != '-' or end_date[7] != '-' or len(end_date) != 10:
                raise('Error: end_date not in correct format (YYYY-MM-DD),' +
                    'end_date is' + str(end_date))
            tweetCriteria = tweetCriteria.setUntil(end_date)
        tweets = got.manager.TweetManager().getTweets(tweetCriteria,halt_count=halt_count)
        if count == 0:
            # extract tweets into data frame
            tweet_props = ['id', 'permalink', 'username', 'text', 'date', 'retweets',
                'favorites', 'hashtags', 'mentions', 'geo', 'keyword']
            tweet_df = pd.DataFrame(index=range(len(tweets)),columns = tweet_props)
            for i,t in enumerate(tweets):
                tweet_df.loc[i, 'id'] = int(t.id)
                tweet_df.loc[i, 'permalink'] = t.permalink
                tweet_df.loc[i, 'username'] = t.username
                tweet_df.loc[i, 'text'] = t.text
                tweet_df.loc[i, 'date'] = t.date
                tweet_df.loc[i, 'retweets'] = t.retweets
                tweet_df.loc[i, 'favorites'] = t.favorites
                tweet_df.loc[i, 'hashtags'] = t.hashtags.replace('# ','#')
                tweet_df.loc[i, 'mentions'] = t.mentions.replace('@ ','@')
                tweet_df.loc[i, 'geo'] = t.geo
                tweet_df.loc[i, 'keyword'] = word
            mark = len(tweet_df.index)
        else:
            for i,t in enumerate(tweets):
                tweet_df.loc[i + mark, 'id'] = int(t.id)
                tweet_df.loc[i + mark, 'permalink'] = t.permalink
                tweet_df.loc[i + mark, 'username'] = t.username
                tweet_df.loc[i + mark, 'text'] = t.text
                tweet_df.loc[i + mark, 'date'] = t.date
                tweet_df.loc[i + mark, 'retweets'] = t.retweets
                tweet_df.loc[i + mark, 'favorites'] = t.favorites
                tweet_df.loc[i + mark, 'hashtags'] = t.hashtags.replace('# ','#')
                tweet_df.loc[i + mark, 'mentions'] = t.mentions.replace('@ ','@')
                tweet_df.loc[i + mark, 'geo'] = t.geo
                tweet_df.loc[i + mark, 'keyword'] = word
            mark = len(tweet_df.index)
    tweet_df = tweet_df.drop_duplicates('id')

    # extra steps to take, assuming we have gotten at least one tweet
    if tweet_df.empty:
        print('No tweets found')
        # return an empty official API dataframe
        api_df = pd.DataFrame(columns=['contributors', 'coordinates',
        'created_at', 'display_text_range', 'entities', 'favorite_count',
        'favorited', 'full_text', 'geo', 'id', 'id_str', 'in_reply_to_screen_name',
        'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id',
        'in_reply_to_user_id_str', 'is_quote_status', 'lang', 'place',
        'possibly_sensitive', 'quoted_status', 'quoted_status_id',
        'quoted_status_id_str', 'quoted_status_permalink', 'retweet_count',
        'retweeted', 'source', 'truncated', 'user', 'permalink',
        'replying_users', 'num_replies'])
        return(api_df)

    if not tweet_df.empty:

        # save link column, which isn't in the official data pull
        link_col = tweet_df[['id','permalink']].copy()
        print('getting official API data')
        # get official API data on the tweets, and work with that
        tweet_df = get_API_data(tweet_df['id'].tolist(), get_retweets = get_retweets)
        tweet_df = tweet_df.merge(link_col, on='id')

        # take quoted statuses, and append to end of the data set if there are any
        if 'quoted_status' in tweet_df.columns:
            clean_stats = tweet_df.quoted_status.dropna()

            # create a dataframe from the list of dictionaries
            for i, chunk in enumerate(clean_stats):
                if i == 0:
                    stats_df = pd.DataFrame(pd.Series(chunk)).transpose()
                else:
                    stats_df = stats_df.append(pd.DataFrame(pd.Series(chunk)).transpose(), ignore_index=True, sort= False)

            # finally, add to the original dataframe of tweets
            tweet_df = tweet_df.append(stats_df, ignore_index = True, sort = False)
        tweet_df = tweet_df.drop_duplicates('id')

        if get_replies == True:
            # get replies from set of tweets using Selenium to open threads on a web browser
            print('getting replies')
            tweet_df = reply_scrap(tweet_df)

            print('getting "in reply to" tweets')

            # get tweets that are found as in reply to in the original set of tweets
            add_tweets = tweet_df.in_reply_to_status_id_str.unique().tolist()
            if add_tweets != [None]:
                if None in add_tweets:
                    add_tweets.remove(None)
                add_tweets_df = get_API_data(add_tweets, get_retweets = get_retweets)
                tweet_df = tweet_df.append(add_tweets_df, ignore_index=True, sort=False)
        tweet_df = clean_API_output(tweet_df,geocode_loc=geocode_loc)
        return(tweet_df)


def extract_counts(title, keywords, start_date=None, get_replies = False, get_retweets = False):
    tweet_df = extract_tweets(title, keywords, start_date = start_date,
        get_replies = get_replies, get_retweets = get_retweets)
    tweet_counts = len(tweet_df.index) + tweet_df.retweets.sum()
    return(tweet_counts)
