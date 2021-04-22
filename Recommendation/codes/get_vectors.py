# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:10:23 2021

@author: ftppr
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim.downloader
import numpy as np
import string
import scipy.spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn import metrics
import pickle
from datetime import datetime
#################
data = pd.read_csv('../input/netflix_titles.csv')
bak = pd.read_csv('../input/netflix_titles.csv')
def process_text2(data):
    data = data.str.lower()
    data = data.apply(lambda row: word_tokenize(row))
    data = data.apply(lambda row: [word for word in row if word.isalpha() and not word in stopwords.words('english')])
    useless_pos_tags = ['CD', 'MD', 'WRB', 'WP'] # 'RB', 'RBS ', 'RBR', 
    data = data.apply(lambda row: [word for (word, tag) in nltk.pos_tag(row) if tag not in useless_pos_tags])
    data = data.apply(lambda row: [word for word in row if word not in list(string.punctuation) + list("@#$%^&*()_+=-……￥·【】[]|\{}") ])
    lemmatizer=WordNetLemmatizer()
    data = data.apply(lambda row: [lemmatizer.lemmatize(word) for word in row])
    return data
data['words'] = process_text2(data['title'] + ' ' + data['description'])
data['genre'] = process_text2(data['listed_in'])

print("process words and genre finished")

mature = ['TV-MA','R','PG-13','TV-14','TV-PG','PG','NC-17']
general = ['NR',np.nan,"G","UR",'TV-G']
young = ['TV-Y','TV-Y7','TV-Y7-FV']
mapping = {}
for item in young:
    mapping[item] = 'Y'
for item in general:
    mapping[item] = 'G'
for item in mature:
    mapping[item] = 'M'
data['rating'] = bak['rating']
data['rating'] = data['rating'].apply(lambda item: mapping[item])


# # ------- process the cast
# def first_3(str_a):
#     if isinstance(str_a,str):
#         return str_a.lower().split(', ')[:3]
#     else:
#         return []
# data['cast'] = data['cast'].apply(first_3)


# ------- process country
'''
Before: "Norway, Iceland, United States"
After: ["norway", "iceland", "united states"]
'''
def split_comma(str_a):
    if isinstance(str_a,str):
        return str_a.lower().split(', ')[:3]
    else:
        return []
def split_comma_set(str_a):
    if isinstance(str_a,str):
        return set(str_a.lower().split(', '))
    else:
        return set()
    
data['country'] = data['country'].apply(split_comma)   
# bak['country'] = bak['country'].apply(split_comma) 
country_lst = []
for country in data['country']:
    country_lst += country
from collections import Counter
country_summary = dict(Counter(country_lst))
country_summary = dict(sorted(country_summary.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))
tmp = list(country_summary.values())
selected_country = dict([kv for kv in country_summary.items() if kv[1] >= 100])
selected_country_names = selected_country.keys()
print(len(selected_country))
print(selected_country_names)



def change_country_name(lst):
    for i in range(len(lst)):
        if lst[i] not in selected_country_names:
            lst[i] = 'other'
    if len(lst) <= 0:
        lst = ['other']
    return lst

data['country'] = data['country'].apply(change_country_name)
# data['genre'] = data['listed_in'].apply(split_comma)
data['genre'] = data['genre'].apply(lambda row: list(set(row)))

print("nontext features processing finished")

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(sparse_output=True)
data2 = data[['type', 'title','country', 'rating', 'words', 'genre']]
data2 = data2.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(data.pop('genre')),
                index=data.index,
                columns=mlb.classes_))
data2 = data2.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(data2.pop('country')),
                index=data.index,
                columns=mlb.classes_))

data2 = data2.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(data2.pop('rating')),
                index=data.index,
                columns=mlb.classes_))

# data2['type'] = data2['type'].apply(lambda row:row.lower().replace(' ','_'))
data2['type'] = data2['type'].apply(lambda row: 1 if row.lower().find('movie')!=-1 else 0)
print("have converted to one-hot representation")
from gensim.models import word2vec
# sentences = word2vec.Text8Corpus('../input/text8/text8')
# training
# model = word2vec.Word2Vec(sentences, vector_size=100)#, window=5, min_count=1, workers=4)

# model.save('text8.model')
model = word2vec.Word2Vec.load("../model/text8.model")
# model.save('text8-635-100.model')
# model = word2vec.Word2Vec.load("text8-635-100.model")
embeddings = model.wv
#index mapping of the orginal title
indices = pd.Series(data2.index, index = data2['title'].str.lower()).drop_duplicates()
#print(indices)
def cal_emb(data):    
    embs = np.zeros([len(data),embeddings.vector_size])
    unknown = []
    for i in range(len(data)):
        row = (data.iloc[i])#.tolist()[0]
        cnt = 0
        for word in row:
            try:
                embs[i] += embeddings[word]            
                cnt += 1
            except:
                unknown.append(word)   
                pass
        if cnt > 0:
            embs[i] /= cnt
        if i % 100 == 0:
            print("finish the {} rows".format(i))
    print("There are {} unknown words.".format(len(unknown)))
    return embs, unknown
embs_words, unknown_words = cal_emb(data['words'])

mat = np.array(data2.drop(columns=['title','words','genre'])).astype(np.int32)
print("There are {} categorical features.".format(mat.shape[1]))

'''
Verification only
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(a)
print(b)
np.concatenate((a, b), axis=1)
'''
vectors  = np.concatenate((mat,embs_words),axis = 1)
print(vectors.shape)
np.save('../data/vectors.npy',vectors)
vectors = np.load('../data/vectors.npy',allow_pickle=True)

def scale(data):
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled
norm_vectors = scale(vectors)


cosine_sim = 1 - scipy.spatial.distance.cdist(norm_vectors,norm_vectors, metric='cosine')
cosine_sim_2 = metrics.pairwise.cosine_similarity(norm_vectors,norm_vectors)


euclidean_sim = scipy.spatial.distance.cdist(norm_vectors,norm_vectors, metric='euclidean')
#jaccard = scipy.spatial.distance.cdist(embs, embs,  metric='jaccard')
df_635 = pd.read_csv('../input/635_with_tconst.csv')
# indices_635 = pd.Series(df_635.index, index = df_635['title'].str.lower()).drop_duplicates()
indices_635 = [indices[m] for m in df_635['title'].str.lower()]


np.savez('../data/dist.npz', cosine_sim, euclidean_sim)  

# select the columns and rows
cosine_sim_635 = cosine_sim[:,indices_635]
cosine_sim_635 = cosine_sim_635[indices_635,:]

euclidean_sim_635 = euclidean_sim[:,indices_635]
euclidean_sim_635 = euclidean_sim_635[indices_635,:]

np.savez('../data/dist-635.npz', cosine_sim_635, euclidean_sim_635)  


