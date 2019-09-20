from BPCI_monitor_class import BPCI_monitor
import datetime
monitor = BPCI_monitor()
monitor.load_tweets(filename="BPCIA_tweets_raw.csv")
monitor.clean_tweets()
monitor.save_tweets(filename="BPCIA_tweets_clean.csv")
