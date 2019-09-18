from BPCI_monitor_class import BPCI_monitor
import datetime
monitor = BPCI_monitor()
# monitor.load_tweets(filename="BPCIA_tweets_raw.csv")
# monitor.clean_tweets()
# monitor.save_tweets(filename="BPCIA_tweets_clean.csv")
monitor.load_tweets(filename="BPCIA_tweets_clean.csv")
monitor.graph_tweet_arrivals(mark_date=datetime.date(year=2019, month=5, day=1))
# monitor.graph_tweet_arrivals(filename='Tweets containing webinar.png',words_filt=['webinar'])
# monitor.graph_tweet_arrivals(filename='Tweets containing medaxiom.png',words_filt=['medaxiom'])
#monitor.word_count_analysis()
#monitor.identify_named_entities()
#monitor.run_pyMABED(nevents=10,theta=.6)
#monitor.load_pyMABED_output()
#monitor.display_pyMABED()
#monitor.score_sentiment()
monitor.gen_word_cloud()
#monitor.apply_TM()
#monitor.create_USA_heat_map()
