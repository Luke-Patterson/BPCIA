from BPCI_monitor_class import BPCI_monitor

monitor = BPCI_monitor()

monitor.extract_tweets(get_retweets=True,geocode_loc=True,start_date='2019-09-11')
monitor.save_tweets(filename="BPCIA_tweets_raw.csv",append=True)
