def extract_tweets_api(title, keywords, start_date=None, search = 'standard', pg_count=100):
    import pandas as pd
    import numpy as np
    from TwitterAPI import TwitterAPI
    from TwitterAPI import TwitterPager
    import datetime
    import json
    #from reply_scraping import reply_scrap

    consumer_key='lu4NOuUBJoqddUXqQbMEwqWLR'
    consumer_secret='gjydYC0PTbWnNZKrN4bscGr1hUcDS6V5NlzPU3n8n3wUciG2Z8'
    access_token_key='1038416706450935808-ahzZtl4lrS1kNjIvgBwCnwsOxKqR1y'
    access_token_secret='oy8TZTD7w7RWwFmuOo3SLPjDFyf6XOfMGXxJ7FJoz2CKo'
    api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)
    if search == 'standard':
        endpoint = 'search/tweets'
    if search == '30day':
        endpoint = 'tweets/search/30day/:Testing'
    if search == 'fullarchive':
        endpoint = 'tweets/search/fullarchive/:Testing'

    start = datetime.datetime.now()
    count = 0
    for word in keywords:
        print("searching keyword: " + word)
        if search == 'standard':
            pager = TwitterPager(api, endpoint, {'q':word, 'count':pg_count,
                'lang':'en', 'tweet_mode':'extended'})
        else:
            if start_date == None:
                start_date = "200603210000"
            pager = TwitterPager(api, endpoint, {'query':word,
                'maxResults':pg_count,'fromDate':start_date})
        for payload in pager.get_iterator(wait=1):
            if count == 0:
                # extract tweets into data frame
                tweet_df = pd.DataFrame(pd.Series(payload)).transpose()
                tweet_df['keyword'] = word
                count = count + 1
                print('getting tweet page ' + str(count))
            else:
                row = pd.Series(payload)
                row['keyword'] = word
                tweet_df = tweet_df.append(row, ignore_index=True, sort=False)
    tweet_df = tweet_df.drop_duplicates('id')

    # obtain tweets in reply to

    # add number of replies
    #tweet_df = reply_scrap(tweet_df)
    return(tweet_df)

# API counts extraction is not very good since you only get 1 month/request
# will be using GoT method for approximation of counts I think

def extract_counts_api(title, keywords, search,start_date=None):
    import pandas as pd
    import numpy as np
    from TwitterAPI import TwitterAPI
    from TwitterAPI import TwitterPager
    import datetime
    import json
    consumer_key='lu4NOuUBJoqddUXqQbMEwqWLR'
    consumer_secret='gjydYC0PTbWnNZKrN4bscGr1hUcDS6V5NlzPU3n8n3wUciG2Z8'
    access_token_key='1038416706450935808-ahzZtl4lrS1kNjIvgBwCnwsOxKqR1y'
    access_token_secret='oy8TZTD7w7RWwFmuOo3SLPjDFyf6XOfMGXxJ7FJoz2CKo'
    api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)
    if search == '30day':
        endpoint = 'tweets/search/30day/:Testing/counts'
    if search == 'fullarchive':
        endpoint = 'tweets/search/fullarchive/:Testing/counts'

    start = datetime.datetime.now()
    count = 0
    for word in keywords:
        print("searching keyword: " + word)
        if start_date == None:
            start_date = "200603210000"
        result = api.request(endpoint, {'query':word, 'fromDate':start_date, 'bucket':'day'})
    return(pd.Series(result.json()['results']))
