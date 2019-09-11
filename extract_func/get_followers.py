import tweepy
import pandas as pd
from ast import literal_eval
import datetime
import time
import math
# function to get followers for a list of user ids
def get_followers(screen_names):
    # query twitter for followers of each unique node:
    for h,i in enumerate(unique_nodes[33:]):
        print('user ' + i + ': ' + str(h) + ' of ' + str(len(unique_nodes)))
        followers=[]
        users = tweepy.Cursor(api.followers, screen_name=i,count=5000).items()
        count = 0
        while True:

            try:
                user = next(users)
                followers.append(user)
            except tweepy.RateLimitError:
                print('hit rate limit for followers endpoint, waiting 15 min from ' + str(datetime.datetime.now()))
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

        if h == 0:
            follow_df = pd.DataFrame([[i,followers]])
        else:
            follow_df = follow_df.append([[i,followers]], ignore_index=True)
    follow_df.columns = ['user', 'followers']
    return(follow_df)

# function to estimate time needed to scrap ids accounting for API rate limits
def follower_time_est(follower_nums):
    curr_time = datetime.datetime.now()
    est_time = curr_time
    est_time += datetime.timedelta(minutes= +len(follower_nums))
    multi_blocks = [math.floor(i/5000) for i in follower_nums]
    est_time += datetime.timedelta(minutes= +sum(multi_blocks))
    # feed a list of follower numbers
    print('Estimated time completed by: ' + str(est_time))
    print('Estimated time required: ' + str(est_time - curr_time))
