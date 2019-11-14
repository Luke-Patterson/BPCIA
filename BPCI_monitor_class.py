# define class of a "monitoring object"
# some custom functions for tweet extraction and analysis
from extract_func.extract_tweets import extract_tweets
from analysis_func.sentiment_analysis import sentiment_scoring
from analysis_func.build_msg_net import build_msg_net
from pyvis.network import Network
import pyvis
# some custom functions for topic modeling
from tm_gui.GUI import run_lda
from tm_gui.GUI import Settings
# basic python packages to import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import wordcloud
from dateutil import parser
import time
import subprocess
import re
# geocoding tools
import geopy
import geopandas as gpd
from geopy.geocoders import Nominatim
geopy.geocoders.options.default_user_agent = "my-application"
from shapely.geometry import Point
# mapping tools
import gmplot
import folium
# data storage libraries
import pickle
import json
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
# entity recognition library
import spacy
# nltk tools for NLP preprocessing
from nltk.corpus import opinion_lexicon as ol
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

class BPCI_monitor:
    """
    This class is a storage object for monitoring and analysis of BPCI tweets.

    """

    def __init__(self):
        pass

    # ====================================================================
    # Data extraction/cleaning functions
    # ====================================================================
    # extract tweets from twitter
    def extract_tweets(self,start_date='2018-01-09',geocode_loc=False, get_retweets=False):
        self.tdf=extract_tweets('BPCI',['BPCIAdvanced','"BPCI Advanced"','"BPCIA"',
            '"BPCI-A"','"Bundled Payments for Care Improvement–Advanced"',
            '"Bundled Payments for Care Improvement Advanced"'],start_date,
           None, get_replies = False, get_retweets = get_retweets, geocode_loc=geocode_loc)
        if geocode_loc:
            self.geocode_tweets()

    # save class' tdf to output folder
    def save_tweets(self,filename,append=False):
        # option to append to existing file
        if append:
            # load existing tweets database
            df= pd.read_csv("output/"+filename)
            df= df.append(self.tdf, ignore_index=True, sort=False)
            # drop duplicate tweet ids that are already present in the csv
            df=df.drop_duplicates(subset='id')
            df=df.sort_values('created_at',ascending=False)
            df.to_csv("output/"+filename,index=False)
            self.tdf=df
        else:
            self.tdf.to_csv("output/"+filename,index=False)

    # load existing tweets from a csv file
    def load_tweets(self,filename):
        self.tdf=pd.read_csv("output/"+filename)
        # make sure date is a timestamp column
        if 'date' in self.tdf.columns:
            self.tdf['date']=self.tdf.date.apply(lambda x: pd.Timestamp(parser.parse(x)))

    # geocode tweets
    def geocode_tweets(self):
        # option to geocode tweets
        # geocode location of user
        geolocator = Nominatim()
        def geo_locate(loc):
            time.sleep(1)
            count=0
            while True:
                try:
                    if pd.isna(loc)==False:
                        geocode=geolocator.geocode(loc)
                        if geocode!=None:
                            return(str(geocode.point.latitude)+', '+str(geocode.point.longitude))
                    break
                except (geopy.exc.GeocoderTimedOut, geopy.exc.GeocoderQuotaExceeded, geopy.exc.GeocoderServiceError):
                    if count<20:
                        time.sleep(1)
                        print('geocoding timeout, retrying attempt #'+str(count))
                        count+=1
                        continue
                    else:
                        raise('Error: too many time out attempts')

        self.tdf['author_geocoord']=self.tdf['author_location'].apply(geo_locate)

    # clean tweet data set
    def clean_tweets(self):
        # df object will be shorthand for self.tdf
        df=self.tdf.copy()
        # twitter extraction function sometimes still includes tweets that don't
        # have any of the listed keywords, unclear why exactly. Will re-check to
        # ensure at least one of our keywords appears in the text of the tweet,
        # otherwise we will drop the tweet
        keywords=['BPCI','Bundled']
        df=df.rename({'full_text':'text','created_at':'date'},axis=1)
        df['text']=df['text'].fillna('')
        for i in keywords:
            df[i]=df.text.apply(lambda x: i.lower() in x.lower())
        df['any_kword']=df[keywords].sum(axis=1)
        df=df.loc[df.any_kword>0]
        df=df.drop(keywords+['any_kword'],axis=1)

        # parse created_at as a datetime object
        df['date']=df['date'].apply(lambda x: pd.Timestamp(parser.parse(x)))
        # create a separate text column for topic model; removing some common
        # terms for a quick cleaning. Probably can be handled better by
        # TF-IDF application in the TM if not already done

        df['filt_text']=df['text'].str.lower().str.replace('bpci', ' ') \
            .str.replace('bundled', ' ') \
            .str.replace('bpcia', ' ') \
            .str.replace('payments', ' ') \
            .str.replace('payment', ' ') \
            .str.replace('care', ' ') \
            .str.replace('improvement', ' ') \
            .str.replace('bpci-a', ' ') \
            .str.replace('advanced', ' ') \
            .str.replace('advance', ' ') \
            .str.replace('cms', ' ')
        # make sure we only have english tweets
        df=df.loc[(df.lang!='es') & (df.lang!='ja')]
        # drop geocoordinates that we know are bad
        df.loc[df['author_location']=='USA','author_geocoord']=np.nan
        # for some reason geocoder sticks these guys in the middle of the USA
        df.loc[df['author_geocoord']=='39.7837304, -100.4458825','author_geocoord']=np.nan
        # drop tweets that are related to a different BPCIA acronym:
        # Biologics Price Competition and Innovation Act of 2009 (enacted in 2010)
        df=df.loc[df['text'].apply(lambda x: 'biologic' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'biosimilar' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'biosims' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'passed in 2010' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'hatch-waxman' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'patent' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'circuit' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'court' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'FDA' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'lawsuit' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'statelaw' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'state law' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'manufacturing' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'litigation' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'trade secret' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'legal jeopardy' not in x.lower())]

        # parse out longitude and latitude
        parse_coord=df['author_geocoord'].str.split(', ',expand=True)
        df['latitude']=parse_coord[0].apply(pd.to_numeric, errors='coerce')
        df['longitude']=parse_coord[1].apply(pd.to_numeric, errors='coerce')
        # flag which tweets contain a keyword indiciating that a vendor is selling something
        def _detect_vendor(str):
            s=str.lower()
            if 'webinar' in s or ' demo' in s or 'tools' in s or 'check out' in s \
                or 'discuss' in s or 'find out' in s or 'learn' in s:
                return(True)
            else:
                return(False)

        df['vendor_flag']= df['text'].apply(_detect_vendor)
        df=df.sort_values('date', ascending=False)

        self.tdf=df



    # ====================================================================
    # Data analysis functions
    # ====================================================================
    # make a histogram of tweet arrivals
    def graph_tweet_arrivals(self,filename='tweets_over_time.png',
        start_date=datetime.date(year = 2018, month = 1, day = 9),
        mark_date=datetime.date(year = 2018, month = 1, day = 9),
        end_date=datetime.date.today(),
        words_filt=None):
        '''
        params:
        filename: name of filename to save tweet graph as picture
        start_date, end_date: range of dates to graph
        mark_date: date to change colors of bars
        words_filt: either None or list. if list, filter to only tweets that
        contain one of the words in the list.
        '''

        # filter to specified time range
        temp_df=self.tdf.copy()
        if words_filt!=None:
            temp_df=temp_df.loc[temp_df.text.apply(lambda x: any([i in x.lower() for i in words_filt]))]
        if temp_df.shape[0]==0:
            print('no tweets found for', words_filt)
            import pdb; pdb.set_trace()
            return
        start_date=pd.Timestamp(start_date)
        end_date=pd.Timestamp(end_date)
        temp_df=temp_df.loc[(temp_df['date']>=start_date) &
            (temp_df['date']<=end_date)]
        # make 1 bin/week, depending on current date
        wks_elapsed=round((end_date-start_date).days/7)
        # note mark date for "recent" changes
        if mark_date!=None:
            mark_date=pd.Timestamp(mark_date)
            mark_wks=round((mark_date-start_date).days/7)
        else:
            mark_wks=wks_elapsed
        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        N, bins, patches =ax.hist(temp_df['date'], bins=wks_elapsed)
        # demarcate data before and after mark date with colors
        for i in range(0,mark_wks):
            patches[i].set_facecolor('#8a8a8a')
        for i in range(mark_wks,wks_elapsed):
            patches[i].set_facecolor('#32a852')
        plt.ylabel('Number of Tweets per week')
        plt.xlabel('Date')
        #plt.legend(['Tweets before ' + str(mark_date)])
        #plt.legend(['Tweets after ' + str(mark_date)])
        # add in lines for key milestones
        ann_df=pd.read_excel('input/BPCI_announcements.xlsx')
        for i,row in ann_df.iterrows():
            # draw a line by the date
            plt.axvline(x=row['Date'],color='#6c0000')
            # add text describing announcement, moving the text down by 10% of
            # the graph to prevent overlapping text
            plt.text(x=row['Date']+datetime.timedelta(days=5),
                y=plt.ylim()[1]-(i*.1+.05)*(plt.ylim()[1]),
                s='<-'+str(row['Date'].date())+': '+row['Announcement'],
                bbox=dict(boxstyle='square', fc='white', ec='none'))
        plt.tight_layout()
        plt.savefig('output/'+filename)
        # save necessary dataframes for Mike to reconstruct
        # CSV No. 1 - number of tweets each week
        bar_df=pd.DataFrame([[mdates.num2date(i),j] for i,j in zip(bins,N)], columns=['Date','Number of Tweets'])
        bar_df.to_csv('output/'+filename+'_bar_values.csv')
        # CSV No. 2 - BPCI_announcements
        # just saving input/BPCI_annonucements.csv
        ann_df.to_csv('output/BPCI_announcements.csv')

    # analysis of word counts over time
    def word_count_analysis(self):
        # store word count results in chunks by week
        wc_df=pd.DataFrame()
        start_date=pd.Timestamp(datetime.date(year = 2018, month = 1, day = 9))
        end_date=pd.Timestamp(datetime.date.today())
        wks_elapsed=round((end_date-start_date).days/7)
        start_week=start_date
        agg_df=pd.DataFrame()
        print('aggregating week counts')
        for i in range(wks_elapsed):
            end_week=pd.Timestamp(start_week+datetime.timedelta(days=7))
            # for each week, get tweets from that week
            temp_df= self.tdf.loc[(self.tdf.date>=start_week)&(self.tdf.date<end_week)]
            # append to agg_df if there are any tweets in the week
            if temp_df.shape[0]!=0:
                # merge into a single string  text from all tweets
                corpus=temp_df.text.str.cat(sep=' ').split()
                # stem and lemmaize text
                stemmer = SnowballStemmer("english")
                corpus = [' '.join([stemmer.stem(word) for word in x.split()]) for x in corpus]
                corpus = [re.sub(r'[^a-zA-Z\s]', ' ', x) for x in corpus]
                # remove hyperlink tokens
                corpus = [i for i in corpus if 'https' not in i]
                corpus = [' '.join(corpus)]
                count_vectorizer = CountVectorizer(stop_words='english')#max_df=0.8, min_df=0.01,
                word_counts = count_vectorizer.fit_transform(corpus)
                word_df=pd.DataFrame(word_counts.toarray(),index=[str(start_week)[0:10]],columns=count_vectorizer.get_feature_names())
                # normalize to proportion of all tweets
                word_df=word_df.fillna(0)
                word_df=word_df/temp_df.shape[0]
                agg_df=agg_df.append(word_df,sort=False)
            # increment start_week
            start_week+=datetime.timedelta(days=7)
        agg_df=agg_df.fillna(0)
        #create df that shows week to week change in word use
        # to reduce noise, convert all to minimum use over past 5 weeks
        min_df=pd.DataFrame(index=agg_df.index,columns=agg_df.columns)
        print('calculating rolling minimums')
        for i in agg_df.columns:
            min_df[i]=agg_df[i].rolling(5).min()
        #create df to flag anomalous changes in those cells that pass a certain minimum
        filt_df=agg_df.copy()
        filt_df=filt_df[(min_df!=0)&(min_df.isna()==False)]
        # drop all missing columns
        filt_df=filt_df.dropna(how='all',axis=1)
        # repopulate remaining columns with all values
        filt_df=agg_df[filt_df.columns].copy()
        # sense of number of non-zero cells picked up this way
        min_df=min_df.iloc[4:,:]
        print('number of non zero agg_df cells')
        print((agg_df.iloc[4:,:].values!=0).sum(axis=1).sum())
        print('number of non zero min_df cells')
        print((min_df.values!=0).sum(axis=1).sum())
        # calculate moving average of word use
        print('calculating rolling averages for those words')
        for i in filt_df.columns:
            filt_df[i]=filt_df[i].rolling(5,min_periods=1).mean()
        chg_df=filt_df.diff().dropna(how='all',axis=0)
        # plot histogram of all values to get a sense of variation in weekly changes in moving average
        vals= chg_df.values.tolist()
        # flatten list of values
        vals=[y for x in vals for y in x]
        plt.clf()
        plt.hist(vals)
        #plt.show()
        plt.clf()
        print('summary statistics of change values')
        print(pd.Series(vals).describe())
        # flag changes that are >10% changes in moving average of normalized word freq in either direction
        flg_df=(chg_df>.1).astype('int')
        temp_flg=(chg_df<-.1).astype('int')
        temp_flg=temp_flg.replace(1,-1)
        flg_df[temp_flg==-1]=-1
        # set filt_df index to datetime so we can graph it
        filt_df.index=[parser.parse(i) for i in filt_df.index]
        # look at which words have "significant" shifts
        sig_words=flg_df.abs().sum()
        sig_words=sig_words.loc[sig_words!=0]
        print([i for i in sig_words.index])
        # running into same problem as topic model; trends in individual words are not particularly meaningful
        # will output weighted average use of words flagged as meaningful data
        ann_df=pd.read_excel('input/BPCI_announcements.xlsx')
        for i in sig_words.index:
            plt.clf()
            filt_df[i].plot(figsize=(10,6))
            plt.xlabel('Date')
            plt.ylabel('% of Tweets containing word stem')
            plt.title('Tweets using word stem: '+ i.title())
            plt.tight_layout()
            # add in lines for key milestones
            for j,row in ann_df.iterrows():
                # draw a line by the date
                plt.axvline(x=row['Date'],color='#6c0000')
            plt.savefig('output/word_trends/'+i+'.png')
        import pdb; pdb.set_trace()

    # Spacy does not seem to do a very good job parsing out entities
    # identification of named entities in tweets
    # def identify_named_entities(self):
    #     # use spacy to parse out entities
    #     nlp = spacy.load("en_core_web_sm")
    #     ent_counts={}
    #     ent_labels=[]
    #     # loop for each tweet containing vendor flag
    #     df=self.tdf.loc[self.tdf['vendor_flag']]
    #     print('identifying entities in ' +str(df.shape[0]) + ' tweets')
    #     for n,i in enumerate(df.text):
    #         if n % 100 ==0:
    #             print(n)
    #         doc = nlp(i)
    #         for ent in doc.ents:
    #             # only add if label is an "org"
    #             if ent.label_=='ORG':
    #                 ent_labels.append([ent.text,ent.label_])
    #     ent_df=pd.DataFrame(ent_labels)
    #     # curate some of the list to exclude misinterpreted entities
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'BPCI' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'Bundled' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'Medicare' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'MIPS' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'MACRA' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'ACO' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'CMMI' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'EHR' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'QPP' not in x)]
    #     ent_df=ent_df.loc[ent_df[0].apply(lambda x: 'Federal Circuit Affirms Noninfringement' not in x)]
    #
    #     # export the tabulations of named entities
    #     ent_df[0].value_counts().to_csv('output/working/entity_tabulation.csv',header=False)
    #
    #     # export those tweets which are flagged as "vendor tweets"
    #     df.to_csv('output/working/vendor_tweets.csv',index=False)
    #
    #     # from this tabulation, manually curate the major vendors identified
    #     # Also, curate based on the list of tweets that have keywords
    #     # indicating its selling something (the vendor_flag column)
    #     curate_df=pd.read_excel('output/working/vendor_entity_list.xlsx')

    # ====================================================================
    # pyMABED functions
    # ====================================================================
    # haven't been able to get meaningful results with pyMABED really
    # Tweet volume anomaly detection - assess whether recent tweet volume (from
    # mark date to present) is anomalous
    # using pyMABED package - cloned/tweeked from https://github.com/AdrienGuille/pyMABED
    def run_pyMABED(self,nevents=10,theta=.6):
        # built to run on command line
        subprocess.run('python pyMABED/detect_events.py output/BPCIA_tweets_clean.csv ' + str(nevents) +
         ' --o output/MABED_out.pkl --sw pyMABED/stopwords/twitter_en.txt --sep ,'
         +' --tsl 1440 --t ' + str(theta))
    # take pyMABED results and visualize them
    def display_pyMABED(self):
        # built to run on command line
        subprocess.run('python pyMABED/build_event_browser.py ' +
        'output/MABED_out.pkl') #--o output/mabed_viz
    # load MABED output pickle
    def load_pyMABED_output(self):
        with open('output/MABED_out.pkl', 'rb') as input_file:
            mabed= pickle.load(input_file)
        event_descriptions = []
        impact_data = []
        formatted_dates = []
        for i in range(0, mabed.corpus.time_slice_count):
            formatted_dates.append(mabed.corpus.to_date(i))
        for event in mabed.events:
            mag = event[0]
            main_term = event[2]
            raw_anomaly = event[4]
            formatted_anomaly = []
            time_interval = event[1]
            related_terms = []
            for related_term in event[3]:
                related_terms.append(related_term[0]+' ('+str("{0:.2f}".format(related_term[1]))+')')
            event_descriptions.append((mag,
                                       str(mabed.corpus.to_date(time_interval[0])),
                                       str(mabed.corpus.to_date(time_interval[1])),
                                       main_term,
                                       ', '.join(related_terms)))
            for i in range(0, mabed.corpus.time_slice_count):
                value = 0
                if time_interval[0] <= i <= time_interval[1]:
                    value = raw_anomaly[i]
                    if value < 0:
                        value = 0
                formatted_anomaly.append('['+str(formatted_dates[i])+','+str(value)+']')
            impact_data.append('{"key":"' + main_term + '", "values":[' + ','.join(formatted_anomaly) + ']}')
        import pdb; pdb.set_trace()

    # ====================================================================
    # Sentiment Analysis and Topic Modeling functions
    # ====================================================================
    # add sentiment scores to each of the tweets using nltk VADER classifier
    # then export positive and negative tweets to csv
    def score_sentiment(self, neg_csv='Negative_BPCI_tweets',pos_csv='Positive_BPCI_tweets',
        start_date=datetime.date(year = 2018, month = 1, day = 9),
        end_date=datetime.date.today(),
        export_csv=True):
        # filter to specified time range
        temp_df=self.tdf.copy()
        start_date=pd.Timestamp(start_date)
        end_date=pd.Timestamp(end_date)
        temp_df=temp_df.loc[(temp_df['date']>=start_date) &
            (temp_df['date']<=end_date)]
        temp_df=sentiment_scoring(temp_df,text_var='text')
        sent_df = temp_df.copy()
        if export_csv:
            # export unsorted values
            sent_df.to_excel('output/BPCI_tweets_sentiment_scores.xlsx', index=False)
            sent_df.to_csv('output/BPCI_tweets_sentiment_scores.csv', index=False)
            # # pull out complaints by looking at negative sentiment scores
            sent_df = sent_df.sort_values('neg',ascending=False)
            #df = df.loc[df['neg']>.1]
            sent_df.to_excel('output/'+neg_csv+'.xlsx', index=False)
            sent_df.to_csv('output/'+neg_csv+'.csv', index=False)
            sent_df = sent_df.sort_values('pos',ascending=False)
            #df = df.loc[df['neg']>.1]
            sent_df.to_excel('output/'+pos_csv+'.xlsx', index=False)
            sent_df.to_csv('output/'+pos_csv+'.csv', index=False)
            self.sent_df=sent_df
        else:
            return(sent_df)

    # function to graph overall sentiment scores over time
    def graph_sentiment(self):
        df=self.score_sentiment(export_csv=False)
        # store sentiment in chunks by week
        start_date=pd.Timestamp(datetime.date(year = 2018, month = 1, day = 9))
        end_date=pd.Timestamp(datetime.date.today())
        wks_elapsed=round((end_date-start_date).days/7)
        start_week=start_date
        agg_df=pd.DataFrame()
        print('aggregating week counts')
        for i in range(wks_elapsed):
            end_week=pd.Timestamp(start_week+datetime.timedelta(days=7))
            # for each week, get tweets from that week
            temp_df= df.loc[(df.date>=start_week)&(df.date<end_week)]
            wk_pos=temp_df.pos.mean()
            wk_neg=temp_df.neg.mean()
            wk_neu=temp_df.neu.mean()
            wk_compound=temp_df['compound'].mean()
            agg_df=agg_df.append(pd.Series([wk_pos,wk_neg,wk_neu,wk_compound],
                name=str(start_week)[0:10]),sort=False)
            # increment start_week
            start_week+=datetime.timedelta(days=7)
        agg_df.columns=['Positive','Negative','Neutral','Compound']
        agg_df.index=[pd.Timestamp(i) for i in agg_df.index]
        agg_df.to_csv('output/sentiment_values_table.csv')
        compar_df=agg_df[['Positive','Negative','Neutral']]
        compound_df=agg_df[['Compound']]
        plt.clf()
        compar_df.plot(figsize=(10,6))
        plt.xlabel('Date')
        plt.ylabel('Mean Score')
        plt.title('Weekly Mean Sentiment Scores')
        # add in lines for key milestones
        ann_df=pd.read_excel('input/BPCI_announcements.xlsx')
        for i,row in ann_df.iterrows():
            # draw a line by the date
            plt.axvline(x=row['Date'],color='#6c0000')
            # add text describing announcement, moving the text down by 10% of
            # the graph to prevent overlapping text
            # plt.text(x=row['Date']+datetime.timedelta(days=5),
            #     y=plt.ylim()[1]-(i*.1+.05)*(plt.ylim()[1]),
            #     s='<-'+str(row['Date'].date())+': '+row['Announcement'],
            #     bbox=dict(boxstyle='square', fc='white', ec='none'))
        plt.savefig("output/sentiment_comparison.png")
        plt.clf()
        compound_df.plot(figsize=(10,6))
        plt.xlabel('Date')
        plt.ylabel('Mean Score')
        plt.title('Weekly Compound Sentiment Score')
        # # add in lines for key milestones
        # for i,row in ann_df.iterrows():
        #     # draw a line by the date
        #     plt.axvline(x=row['Date'],color='#6c0000')
        #     # add text describing announcement, moving the text down by 10% of
        #     # the graph to prevent overlapping text
        #     plt.text(x=row['Date']+datetime.timedelta(days=5),
        #         y=plt.ylim()[1]-(i*.1+.05)*(plt.ylim()[1]),
        #         s='<-'+str(row['Date'].date())+': '+row['Announcement'],
        #         bbox=dict(boxstyle='square', fc='white', ec='none'))
        plt.savefig("output/sentiment_compound.png")
    # generate word clouds from text of tweets, positive and negative
    def gen_word_cloud(self):
        raw_text = ' '.join(self.tdf.filt_text.to_list())
        wc = wordcloud.WordCloud().process_text(raw_text)
        # assign postive/negative connotations to words from nltk corpus
        sent_df = pd.DataFrame(columns=['Words','align'])
        print(len(wc.keys()))
        for h,i in enumerate(wc.keys()):
            if h % 10 ==0:
                print(h)
            sent_df.loc[h,'Words']= i
            if i in ol.positive():
                sent_df.loc[h,'align']= 'Positive'
            elif i in ol.negative():
                sent_df.loc[h,'align']= 'Negative'
            else:
                sent_df.loc[h,'align']= 'Neutral'
        pos_words = sent_df.loc[sent_df['align']== 'Positive', 'Words'].tolist()
        neg_words = sent_df.loc[sent_df['align']== 'Negative', 'Words'].tolist()

        pos_dict =  {k:v for k,v in wc.items() if k in pos_words}
        neg_dict =  {k:v for k,v in wc.items() if k in neg_words}

        wordcloud.WordCloud(width=800,height=400).generate_from_frequencies(pos_dict).recolor(colormap='Greens').to_file('output/pos_wordcloud.png')
        wordcloud.WordCloud(width=800,height=400).generate_from_frequencies(neg_dict).recolor(colormap='Reds').to_file('output/neg_wordcloud.png')

    # run tweet language through LDA topic model
    def apply_TM(self,in_filename='BPCIA_tweets_clean',out_results='TM_results',
        out_topics='TM_topics'):
        settings=Settings()
        settings.file_name='output/'+in_filename+'.csv'
        results=run_lda(settings)
        # save results
        self.TM_results=results[0]
        self.TM_topics=results[1]
        results[0].to_pickle('output/'+out_results+'.pkl')
        with open('output/'+out_topics+'.json', 'w') as fjson:
             json.dump(results[1], fjson)
        topic_df=pd.read_json('output/'+out_topics+'.json')
        topic_df.to_csv('output/'+out_topics+'.csv')

    # ====================================================================
    # Geospatial Analysis functions
    # ====================================================================
    # create heat map of US-based tweets
    def create_USA_heat_map(self):
        longitudes=self.tdf.loc[self.tdf['longitude'].isna()==False,'longitude']
        latitudes=self.tdf.loc[self.tdf['latitude'].isna()==False,'latitude']
        # need to add personal Google API key below for this to work
        key= 'add Google API Key'
        gmap = gmplot.GoogleMapPlotter(38.817191, -100.068385,5,apikey=key)
        # Overlay our datapoints onto the map
        gmap.heatmap(latitudes, longitudes,radius=75,opacity=.8)
        # Generate the heatmap into an HTML file
        gmap.draw("output/tweet_location_heatmap.html")
        #--------------
        # obsolete code
        #--------------
        # # Set geometry for geo df
        # geometry= [Point(xy) for xy in zip(self.tdf['longitude'],self.tdf['latitude'])]
        # crs = {'init': 'epsg:3857'}
        # geo_df=gpd.GeoDataFrame(self.tdf, crs=crs, geometry=geometry)
        # # load in US shapefile
        # bg = gpd.read_file('input/shp/state/tl_2017_us_state.shp')
        # sts_out = ['AS', 'GU', 'MP', 'VI','HI']
        # bg = bg[~bg['STUSPS'].isin(sts_out)]
        # bg = bg.sort_values(by='STUSPS')
        # ax = bg.plot(color='snow', linewidth=0.3, edgecolor='gray')
        # ax = geo_df.plot(ax=ax, color='k',alpha=.75)
        # plt.savefig('output/tweet_heat_map.png')
    # create tabulation of locations, since not all that many are ready
    def create_location_tabulation(self):
        self.tdf['author_location']=self.tdf.author_location.str.replace(".","")
        locs_tab=self.tdf.author_location.value_counts().dropna()
        locs_tab=locs_tab.loc[[',' in s for s in locs_tab.index]]
        locs_tab=locs_tab.loc[['#' not in s for s in locs_tab.index]]
        locs_tab=locs_tab.loc[locs_tab>10]
        plt.clf()
        plt.bar(locs_tab.index,locs_tab.values)
        plt.xticks(rotation='vertical')
        plt.xlabel('City')
        plt.ylabel('Number of Tweets from City')
        plt.title('Geolocation of Tweets')
        plt.tight_layout()
        plt.savefig('output/location_bar_chart.png')
        # create tabulation weighted by city population
        # load in city population data
        city_df=pd.read_csv('input/ACS_city_pop_2018.csv',encoding='latin-1')
        # clean up city_df geography tag
        city_df['Geography']=city_df['Geography'].str.replace('city','')
        city_df['Geography']=city_df['Geography'].str.replace('town','')
        city_df['Geography']=city_df['Geography'].str.replace('.','')
        geo_split=city_df['Geography'].str.split(',',expand=True)
        city_df['City']=geo_split[0].str.strip()
        city_df['State']=geo_split[1].str.strip()
        # add in state acronym
        state_df=pd.read_csv('input/abbr-name.csv')
        city_df=city_df.merge(state_df, how='left', on='State')
        # clean up some weird values
        city_df.loc[city_df['State']=='District of Columbia','Acronym']='DC'
        city_df.loc[city_df['City']=='Nashville-Davidson metropolitan government (balance)','City']='Nashville'
        # merge in state population
        locs_tab=locs_tab.reset_index()
        locs_tab=locs_tab.rename({'author_location':'num_tweets'},axis=1)
        geo_split=locs_tab['index'].str.split(',',expand=True)
        locs_tab['City']=geo_split[0].str.strip()
        locs_tab['State']=geo_split[1].str.strip()
        locs_tab['State']=locs_tab['State'].str.replace("New York",'NY')
        locs_tab.loc[locs_tab['City']=='Danvers','City']='Beverly'
        locs_tab=locs_tab.merge(city_df[['City','Acronym','2018 Population Estimate']],
            how='left',left_on=['City','State'],right_on=['City','Acronym'])
        # normalize tweets by population
        locs_tab['tweets_per_100k']=locs_tab['num_tweets']*100000/locs_tab['2018 Population Estimate']
        locs_tab=locs_tab.sort_values('tweets_per_100k',ascending=False)
        temp_locs=locs_tab.copy()
        temp_locs=temp_locs.loc[temp_locs['2018 Population Estimate']>100000]
        # redo tabulation with tweets per 100k population
        plt.clf()
        plt.bar(temp_locs['index'],temp_locs.tweets_per_100k)
        plt.xticks(rotation='vertical')
        plt.xlabel('City')
        plt.ylabel('Number of Tweets per 100,000 Population')
        plt.title('Geolocation of Tweets')
        plt.tight_layout()
        plt.savefig('output/location_bar_chart_pop_norm.png')
        # save tabulation as CSV
        locs_tab.to_csv('output/location_tabulation.csv',header=True)
        import pdb; pdb.set_trace()

    # build an interactive visualization of the network
    def network_viz(self):
        # build a networkX graph object from tweet data frame
        self.net_graph=build_msg_net(self.tdf)
        # remove self edges
        self.net_graph.remove_edges_from(self.net_graph.selfloop_edges())
        # start pyviz object
        viz=Network(height=800,width=800)
        viz.from_nx(self.net_graph)
        #viz.show_buttons()
        # open the preset options we want to specifiy
        viz_options=open("input/pyviz_options.txt",'r').read()
        viz.set_options(viz_options)
        # show the graph in browser
        viz.show('tweet_network_viz.html')
        import pdb; pdb.set_trace()
