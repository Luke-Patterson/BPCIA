from wordfreq import word_frequency
import nltk
import math
import os
os.environ['CLASSPATH'] = "C:/Users/lpatterson/AnacondaProjects/twitter_experiment/extract_func/Keyword_generation/stanford-postagger-2018-10-16"
os.environ['JAVAHOME'] = "C:/Program Files (x86)/Java/jre1.8.0_211/bin/java.exe"
os.environ['STANFORD_MODELS'] = "C:/Users/lpatterson/AnacondaProjects/twitter_experiment/extract_func/Keyword_generation/stanford-postagger-2018-10-16/models"
from nltk.tag import StanfordPOSTagger
from nltk.tag import  pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import string

def gen_queries(df,nwords=5,stanford = False):
    df.Title = df.Title.str.replace('-', ' ').str.replace('/', ' ')
    # split words of texts
    split_df = df.Title.str.split(pat=" ", expand=True)

    # clear out punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    def punc_rm(s):
        if s!=None:
            s = str(s).translate(translator).upper()
            s = s.replace('-', ' ')
            s = s.replace('/', ' ')
            s = s.replace('?', '')
            # remove word if weird characters exist
            try:
                s.encode('ascii')
                return(s)
            except:
                pass
    split_df = split_df.applymap(punc_rm)
    split_df = split_df.replace([''],[None])

    # tag part of speech
    split_list = split_df.fillna('None').values.tolist()
    # feed each row separately into the tagger
    part_list = []
    # Keep only nouns/verbs
    if stanford == True:
        tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
        for h,i in split_df.fillna('None').iterrows():
            print(h)
            values = i.values.tolist()
            part_list.append(tagger.tag(values))
    else:
        for h,i in split_df.fillna('None').iterrows():
            values = i.values.tolist()
            part_list.append(pos_tag(values,lang='eng'))
    part_df = pd.DataFrame(part_list)
    part_df1 = part_df.applymap(lambda x: x[0])
    part_df2 = part_df.applymap(lambda x: x[1])

    split_df = pd.DataFrame(np.where(((part_df2 == 'NN') | (part_df2 == 'NNP') |
        (part_df2 == 'NNS') | (part_df2 == 'NNPS')
        # | (part_df2 == 'VB') | (part_df2 == 'VBD')
        # | (part_df2 == 'VBG') | (part_df2 == 'VBN') | (part_df2 == 'VBP') | (part_df2 == 'VBZ')
        ) & (part_df1 != 'None'),split_df,None))

    # remove duplicate words beyond first instance
    dupes = split_df.apply(lambda x: x.duplicated(), axis=1)
    dupes[split_df.isna()] = False
    split_df[dupes] = None
    # scraped: doesn't distinguish all that weell

    # get stems of each word
    # def steming(s):
    #     if s!= None:
    #         stemmer = PorterStemmer()
    #         return(stemmer.stem(s))
    # split_df = split_df.applymap(steming)
    # scraped: twitter doesn't allow partial word search it seems

    # score typical frequency of words
    def freq_str(str):
        if str!=None:
            return(word_frequency(str, 'en'))
        else:
            return(np.nan)
    score_df = split_df.applymap(freq_str)

    # select 3 least common words
    #min_words = score_df.apply(lambda x: x.nsmallest(n=3, keep='all'))
    min_df = pd.DataFrame(0, columns=score_df.columns, index=score_df.index)
    min_df.iloc[:,:] = 0
    for i,j in score_df.iterrows():
        smallest = j.nsmallest(n=nwords, keep='all')
        min_df.loc[i,smallest.index] = 1

    phrase_df = split_df.copy()
    phrase_df = pd.DataFrame(np.where(min_df == 0,None,phrase_df))
    out_df = pd.Series(index=df.index)
    for i,j in phrase_df.iterrows():
        row = ' '.join(j.dropna().reset_index(drop=True).values.tolist())
        out_df.iloc[i] = [row]
    return(out_df)
