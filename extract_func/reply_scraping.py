# for each tweet, retrieve users that replied from twitter permalinks
def reply_scrap(tweet_df):
    import pandas as pd
    import tweepy
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver import ActionChains
    from selenium.webdriver.common.keys import Keys
    import time
    from cursor_iter import cursor_iter

    # set up tweepy object
    consumer_key='lu4NOuUBJoqddUXqQbMEwqWLR'
    consumer_secret='gjydYC0PTbWnNZKrN4bscGr1hUcDS6V5NlzPU3n8n3wUciG2Z8'
    access_token='1038416706450935808-ahzZtl4lrS1kNjIvgBwCnwsOxKqR1y'
    access_token_secret='oy8TZTD7w7RWwFmuOo3SLPjDFyf6XOfMGXxJ7FJoz2CKo'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # start chrome webdriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(executable_path= \
        "C:/Users/lpatterson/AnacondaProjects/chrome_driver/chromedriver.exe")
    actionChains = ActionChains(driver)

    # set starting value of some vars we will use in the iterrows loop
    reply_df_bool = False
    tweet_df['replying_users'] = ''
    tweet_df['num_replies_raw'] = ''
    replying_users  = []
    for i,row in tweet_df.iterrows():
        url = row.permalink
        # for each link, open url
        count = 0
        while True:
            try:
                driver.get(url)
                time.sleep(1)
                break
            except:
                if count <= 20:
                    print('Error opening response tweet URL, attempt #' + str(count))
                    time.sleep(1)
                    count += 1
                    continue
                else:
                    raise('too many attempts to open response tweet URL')
        # OBSOLETE - just going to use favorite_count column from tweet df
        # # get list of liking users
        # try:
        #     num_likes = driver.find_element_by_xpath("//*[@class='ProfileTweet-actionButton js-actionButton js-actionFavorite']//*[@class='ProfileTweet-actionCountForPresentation']").text
        # except:
        #     num_likes = ''
        #
        # if num_likes != '':
        #     driver.find_element_by_xpath("//*[@class='request-favorited-popup']").click()
        #     liking_users = [j.text for j in driver.find_elements_by_xpath("//*[@class='account  js-actionable-user js-profile-popup-actionable ']//*[@class='username u-dir u-textTruncate']")]

        # get number of replies
        try:
            num_replies = driver.find_element_by_xpath("//*[contains(@class, 'focus')]//*[contains(@class, 'js-actionReply')]//*[@class='ProfileTweet-actionCountForPresentation']").text
            print('found ' + num_replies + ' replies for tweet #' + str(i))
        except:
            num_replies = ''

        # actions for when we do find replies
        if num_replies != '':

            # convert the K and M letters to numeric values in num_replies
            def km_convert(str):
                sr_in = pd.Series(str).str.lower()
                sr_out = round(sr_in.replace(r'[km]+$', '', regex=True).astype(float) * sr_in.str.extract(r'[\d\.]+([km]+)', expand=False).fillna('1').replace(['k','m'], [10**3, 10**6]).astype(int))
                return(sr_out)
            num_replies = km_convert(num_replies)

            # if there are replies, scroll until they are all loaded
            # pressing page down once for every ten replies seems to get us to the end
            # issues with very large numbers of replies, will need to truncate
            for j in range(min(round(int(num_replies)/10),20)):
                actionChains.send_keys(Keys.PAGE_DOWN).perform()
                time.sleep(.4)

            # note usernames of those who replied
            replying_users = [j.text for j in
                driver.find_elements_by_xpath("//*[@class='account-group js-account-group js-action-profile js-user-profile-link js-nav']//*[@class='username u-dir u-textTruncate']")]
            replying_users = list(set(replying_users))

            # INCOMPLETE - unsure this is really worth the value of fleshing out
            # # recursively find replies to replies
            # import pdb; pdb.set_trace()
            # lvl_count = 2
            # while True:
            #      print('getting level ' + str(lvl_count) + ' replies')
            #      lvl_count += 1
            #      non_zero

        tweet_df.at[i, 'replying_users'] = replying_users
        tweet_df.at[i, 'num_replies_raw'] = num_replies
        replying_users  = []

        # OBSOLETE - turns out twitter timelines don't include replies
        # for each replying user, get their timeline and look for tweets that are in reply to original tweet
        # for user in replying_users:
        #     print('getting replies from user: ' +  user)
        #     user_tweets = cursor_iter(tweepy.Cursor(api.user_timeline, id=user,count=3200).items())
        #     user_tweets = [i._json for i in user_tweets]
        #     _tweet_df = pd.DataFrame.from_records(user_tweets)
        #     if reply_df_bool == False:
        #         reply_df = _tweet_df.loc[_tweet_df.in_reply_to_status_id == row.id]
        #         reply_df_bool = True
        #     else:
        #         reply_df = reply_df.append(_tweet_df.loc[_tweet_df.in_reply_to_status_id == row.id],
        #         ignore_index=True, sort=False)
        # import pdb; pdb.set_trace()
    tweet_df['num_replies_list'] = tweet_df['replying_users'].apply(lambda x: len(x))

    # close webdriver
    driver.close()

    return(tweet_df)
