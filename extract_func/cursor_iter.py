# function to loop over tweepy cursor items, while handling rate limit and
# other errors
def cursor_iter(cursor):
    # cursor - a tweepy cursor object
    from datetime import datetime
    import time
    import tweepy
    out_list = []
    count = 0
    while True:
        try:
            item = next(cursor)
            out_list.append(item)
        except tweepy.RateLimitError:
            print('hit rate limit, waiting 15 min from ' + str(datetime.now()))
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
                continue
            continue
    return(out_list)
