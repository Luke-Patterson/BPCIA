from ast import literal_eval
import pandas as pd
import numpy as np
import dateutil.parser
from datetime import datetime
import geopy
import time
from geopy.geocoders import Nominatim
geopy.geocoders.options.default_user_agent = "my-application"

# cleaning function to prepare official Twitter API output for analysis functions
def clean_API_output(api_df,geocode_loc):
    def _leval(x):
        try:
            return literal_eval(x)
        except:
            return None

    # transform dict strings to actual dict objects if string
    if all(api_df.entities.apply(type) == str) == False & \
        any(api_df.entities.apply(type) == str) == True:
        raise('mix of strings and dicts in df column "entities"')
    if all(api_df.user.apply(type) == str) == False & \
        any(api_df.user.apply(type) == str) == True:
        raise('mix of strings and dicts in df column "user"')
    if all(api_df.entities.apply(type) == str):
        api_df['entities'] = api_df['entities'].apply(_leval)
    if all(api_df.user.apply(type) == str):
        api_df['user'] = api_df['user'].apply(_leval)

    # add unique user_mentions as separate column
    def _get_nested_dict(dict, key = None, subkey = None, unique = True):
        if dict is not None:
            dict_out = dict[key]
            if subkey is not None:
                dict_out = [i[subkey] for i in dict_out]
            if unique == True:
                dict_out = list(set(dict_out))
            return(dict_out)
        else:
            return(None)
    api_df['user_mentions'] = api_df['entities'].apply(_get_nested_dict,
        args = ('user_mentions','screen_name',True))
    # do the same with hashtags
    api_df['hashtags'] = api_df['entities'].apply(_get_nested_dict,
        args = ('hashtags','text',True))
    # and with author name
    api_df['author_name'] = api_df['user'].apply(_get_nested_dict,
        args = ('screen_name',None,False))
    api_df['author_followers'] = api_df['user'].apply(_get_nested_dict,
        args = ('followers_count',None, False))
    # author  location as well
    api_df['author_location'] = api_df['user'].apply(_get_nested_dict,
        args = ('location',None, False))
    if geocode_loc:
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
                except (geopy.exc.GeocoderTimedOut, geopy.exc.GeocoderQuotaExceeded):
                    if count<20:
                        time.sleep(1)
                        print('geocoding timeout, retrying attempt #'+str(count))
                        count+=1
                        continue
                    else:
                        raise('Error: too many time out attempts')

        api_df['author_geocoord']=api_df['author_location'].apply(geo_locate)

    # standardize created_by date
    def stand_date(date):
        try:
            clean_date = dateutil.parser.parse(date)
            clean_date = clean_date.strftime('%a %b %d %X %Y')
            clean_date = dateutil.parser.parse(clean_date)
            return(clean_date)
        except:
            return(date)
    api_df['created_at'] = api_df['created_at'].apply(stand_date)
    return(api_df)
