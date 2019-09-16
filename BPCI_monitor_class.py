# define class of a "monitoring object"
# some custom functions
from extract_func.extract_tweets import extract_tweets
from analysis_func.sentiment_analysis import sentiment_scoring
# other packages to import
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import datetime
import wordcloud
from tm_gui.GUI import run_lda
from tm_gui.GUI import Settings
from dateutil import parser
import geopy
import geopandas as gpd
import time
from geopy.geocoders import Nominatim
geopy.geocoders.options.default_user_agent = "my-application"
from shapely.geometry import Point
import folium
import pickle
import json
import subprocess
import spacy
import gmplot
from nltk.corpus import opinion_lexicon as ol

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
           None, get_replies = False, get_retweets = get_retweets, geocode_loc=False)
        if geocode_loc:
            self.geocode_tweets()


    # save class' tdf to output folder
    def save_tweets(self,filename,append=False):
        if append:
            # load existing tweets database
            df= pd.read_csv("output/"+filename+".csv")
            df= df.append(self.tdf, ignore_index=True, sort=False)
            # drop duplicate tweet ids that are already present in the csv
            df=df.drop_duplicates(subset='id')
            df.to_csv("output/"+filename,index=False)
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
        df=df.loc[df['text'].apply(lambda x: '2010' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'hatch-waxman' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'patent' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'circuit' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'court' not in x.lower())]
        df=df.loc[df['text'].apply(lambda x: 'FDA' not in x)]

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

    # generate word counts over time

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
        end_date=datetime.date.today()):
        # filter to specified time range
        temp_df=self.tdf.copy()
        start_date=pd.Timestamp(start_date)
        end_date=pd.Timestamp(end_date)
        temp_df=temp_df.loc[(temp_df['date']>=start_date) &
            (temp_df['date']<=end_date)]
        temp_df=sentiment_scoring(temp_df,text_var='text')
        sent_df = temp_df.copy()
        # # pull out complaints by looking at negative sentiment scores
        sent_df = sent_df.sort_values('neg',ascending=False)
        #df = df.loc[df['neg']>.1]
        sent_df.to_csv('output/'+neg_csv+'.csv', index=False)
        sent_df = sent_df.sort_values('pos',ascending=False)
        #df = df.loc[df['neg']>.1]
        sent_df.to_csv('output/'+pos_csv+'.csv', index=False)

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

        wordcloud.WordCloud().generate_from_frequencies(pos_dict).recolor(colormap='Greens').to_file('output/pos_wordcloud.png')
        wordcloud.WordCloud().generate_from_frequencies(neg_dict).recolor(colormap='Reds').to_file('output/neg_wordcloud.png')

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
    # TODO: currently just some dots on USA shapefile, make a prettier heat map with
    # OpenStreetMap backdrop.
    def create_USA_heat_map(self):
        longitudes=self.tdf.loc[self.tdf['longitude'].isna()==False,'longitude']
        latitudes=self.tdf.loc[self.tdf['latitude'].isna()==False,'latitude']
        # Temporarily put in a default Gmap API key for ease of development - remove later
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
