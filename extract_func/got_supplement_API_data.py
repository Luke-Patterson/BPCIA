
# take tweet ids and get data on them from official Twitter API
def get_API_data(tweet_ids, get_retweets):
    import tweepy
    from datetime import datetime
    import time
    import pandas as pd
    import numpy as np
    print('Getting official API data for ' + str(len(tweet_ids)) + ' tweets')
    # set up tweepy object
    consumer_key='lu4NOuUBJoqddUXqQbMEwqWLR'
    consumer_secret='gjydYC0PTbWnNZKrN4bscGr1hUcDS6V5NlzPU3n8n3wUciG2Z8'
    access_token='1038416706450935808-ahzZtl4lrS1kNjIvgBwCnwsOxKqR1y'
    access_token_secret='oy8TZTD7w7RWwFmuOo3SLPjDFyf6XOfMGXxJ7FJoz2CKo'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    # use method to look up statuses in chunks of 100
    id_chunks = [tweet_ids[x:x+100] for x in range(0, len(tweet_ids), 100)]
    tweets = []
    for i, chunk in enumerate(id_chunks):
        count = 0
        print('Getting chunk #' + str(i + 1))
        while True:
            try:
                tweets.append(api.statuses_lookup(chunk,tweet_mode= 'extended'))
                break
            except tweepy.RateLimitError:
                print('hit rate limit for statuses_lookup endpoint, waiting 15 min from ' + str(datetime.now()))
                time.sleep(60*15+1)
            except:
                print('Error encountered, retrying')
                if count < 20:
                    print('attempt #' + str(count))
                    count += 1
                    continue
                else:
                    raise ('too many unsuccessful attempts ')
    for i, chunk in enumerate(tweets):
        if i == 0:
            api_df = pd.DataFrame([i._json for i in chunk])
        else:
            api_df = api_df.append(pd.DataFrame([i._json for i in chunk]), ignore_index=True, sort= False)
    # get retweets if necessary
    if get_retweets == True:
        api_df['retweeters'] = ''
        print('Getting retweeters for ' + str(len(api_df.loc[api_df['retweet_count']>0, 'id_str'])) + ' retweets')
        # store retweeter ids in df
        rt_ser = pd.Series()
        for h,i in enumerate(api_df.loc[api_df['retweet_count']>0, 'id_str']):
            if h % 10 == 0:
                print('gotten retweets for ' + str(h) + ' tweets so far' )
            retweeters = []
            cursor = tweepy.Cursor(api.retweeters, id=i,stringify_ids=True).items()
            count = 0
            while True:
                try:
                    retweeter = next(cursor)
                    retweeters.append(retweeter)
                except tweepy.RateLimitError:
                    print('hit rate limit for retweeters endpoint, waiting 15 min from ' + str(datetime.now()))
                    time.sleep(60*15+1)
                except StopIteration:
                    break
                except:
                    print('Other error encountered, retrying ')
                    if count < 20:
                        print('attempt #' + str(count))
                        count += 1
                        continue
                    else:
                        raise ('too many unsuccessful attempts ')
                    continue
            #api_df.at[(api_df['id_str']== i), 'retweeters'] = {api_df.loc[(api_df['id_str']== i)].index[0]:retweeters}
            # next, look up id's to get screen names
            # getting the retweeter names
            rt_ser.loc[api_df.loc[(api_df['id_str']== i)].index[0]] = get_retweeters_names(retweeters)
        api_df['retweeters'] = rt_ser
        # create retweet followers count column
        api_df['rt_follower_counts'] = api_df.loc[api_df['retweeters'].isna() == False
            ,'retweeters'].apply(lambda x: [i['followers_count'] for i in x])
        api_df['rt_names'] = api_df.loc[api_df['retweeters'].isna() == False
            ,'retweeters'].apply(lambda x: [i['screen_name'] for i in x])

    api_df['created_at'] = api_df['created_at'].apply(lambda x: datetime.strptime(x, '%a %b %d %X +0000 %Y'))
    api_df = api_df.sort_values('created_at',ascending=False)
    return(api_df)

def get_retweeters_names(rt_list):
    import tweepy
    from datetime import datetime
    import time
    import pandas as pd
    consumer_key='lu4NOuUBJoqddUXqQbMEwqWLR'
    consumer_secret='gjydYC0PTbWnNZKrN4bscGr1hUcDS6V5NlzPU3n8n3wUciG2Z8'
    access_token='1038416706450935808-ahzZtl4lrS1kNjIvgBwCnwsOxKqR1y'
    access_token_secret='oy8TZTD7w7RWwFmuOo3SLPjDFyf6XOfMGXxJ7FJoz2CKo'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    count = 0
    id_chunks = [rt_list[x:x+100] for x in range(0, len(rt_list), 100)]
    for i, chunk in enumerate(id_chunks):
        count = 0
        while True:
            try:
                if count == 0:
                    rt_chunks = api.lookup_users(user_ids = chunk)
                else:
                    rt_chunks.append(api.lookup_users(user_ids = chunk))
                break
            except tweepy.RateLimitError:
                print('hit rate limit for users_lookup endpoint, waiting 15 min from ' + str(datetime.now()))
                time.sleep(60*15+1)
            except:
                print('Error encountered, retrying')
                if count < 20:
                    print('attempt #' + str(count))
                    count += 1
                    continue
                else:
                    raise ('too many unsuccessful attempts ')
    # flatten
    if len(id_chunks) > 1:
        rt_flat = [item for sublist in rt_chunks for item in sublist]
    if id_chunks == []:
        return([])
    else:
        rt_flat = rt_chunks
    # concatenate
    rt_flat = [i._json for i in rt_flat]
    return(rt_flat)
