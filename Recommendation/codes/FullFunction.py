# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:48:27 2021

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
from utils import *

data = pd.read_csv('../input/netflix_titles.csv')
# # Use only 635
# data = pd.read_csv('../input/635_with_tconst.csv')
# data = data.dropna(subset=['tconst'])

data = drop_unnamed(data)
data = data.reset_index()
data = data[[ 'tconst','type', 'title', 'director', 'cast', 'country', 'rating', 
             'listed_in', 'description', 'words', 'genre']]
bak = data


data['words'] = process_text2(data['title'] + ' ' + data['description'])
data['genre'] = process_text2(data['listed_in'])

# -------- Get the embeddings
# ----- Modify the text8 for training 
more_sentences = []
words_set = set() # for checking only
for i in range(len(data['words'])):
    tmp = data['words'].iloc[i]
    more_sentences.append(tmp)
    words_set.update(tmp)
str_total = ''
for lst in more_sentences:
    str_add = ' '.join(lst)
    str_total += str_add
file = '../input/text8/text8-bak'

with open(file, 'a+',encoding='utf-8') as f:
    f.write(str_total)

# ----- Train the model
# model.build_vocab(more_sentences)
# model.train(more_sentences,total_examples=len(more_sentences), total_words=len(words_set), epochs=10)

# sentences = word2vec.Text8Corpus('../input/text8/text8')
# training
# model = word2vec.Word2Vec(sentences, vector_size=100)#, window=5, min_count=1, workers=4)
# model.save('text8.model')

# ----- Load the model
model = word2vec.Word2Vec.load("../model/text8.model")
embeddings = model.wv

embs_words, unknown_words = cal_emb(data['words'])
# embs_genre, unknown_genre = cal_emb(data['genre'])

embs_words = scale(embs_words)
# embs_genre = scale(embs_genre)

# -------- Process non-text
# Process rating (target audience)
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


# ------- process country
'''
Before: "Norway, Iceland, United States"
After: ["norway", "iceland", "united states"]
'''

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

# ----------- Convert to One-Hot representation
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

mat = np.array(data2.drop(columns=['title','words','genre'])).astype(np.int32)
print("There are {} categorical features.".format(mat.shape[1]))
vectors  = np.concatenate((mat,embs_words),axis = 1)
print(vectors.shape)
np.save('../data/vectors.npy',vectors)


# ----- Calculate and save the distance 
vectors = np.load('../data/vectors.npy',allow_pickle=True)
norm_vectors = scale(vectors)
cosine_sim = 1 - scipy.spatial.distance.cdist(norm_vectors,norm_vectors, metric='cosine')

euclidean_sim = scipy.spatial.distance.cdist(norm_vectors,norm_vectors, metric='euclidean')

np.savez('../data/dist.npz', cosine_sim, euclidean_sim)  
# np.savez('../data/dist-sub.npz', cosine_sim_genre,cosine_sim_words,euclidean_genre,euclidean_words)  

# -------------- Recommend the most similar k items
def recommend(title, data, metric='cosine'):
    files = np.load('../data/dist-635.npz')
    cosine_sim = files['arr_0']
    euclidean_sim = files['arr_1']
    del files # save memory
    indices = pd.Series(data.index, index = data['title'].str.lower()).drop_duplicates()

    idx = indices[title.lower()]
    if metric =='cosine':
        sim_vec = cosine_sim[idx]
        # Get the pairwsie similarity scores of all movies with that movie
    elif metric == 'euclidean':
        sim_vec = euclidean_sim[idx]
        
    sim_scores = list(enumerate(sim_vec))        
    # Sort the movies based on the similarity scores
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    res = sorted_sim_scores[:11]
    res = [item for item in res if item[0]!=idx]

    # Get the movie indices
    res_indices = [item[0] for item in res]
    res_score = [item[1] for item in res]   
    
    final_data = data.iloc[res_indices]
    final_data.insert(0,'score',res_score)
    
    with_input_indices = [idx]+res_indices
    with_input_score = [1]+res_score 
    
    with_input_data = data.iloc[with_input_indices]
    with_input_data.insert(0,'score',with_input_score)
    col_lst = [ 'score', 'tconst', 'type', 'title',  'country', 'rating',
       'listed_in', 'description']
    return final_data[col_lst], with_input_data[col_lst]


title = 'The Visit'
metric='cosine'
res1, res2, final_data = recommend(title,data, metric)


# -------------- Recommend based on user input
def search_recommend(sentence, data, metric='cosine',w_genre=0.5):
    file = np.load("../data/emb_w_g.npz",allow_pickle=True)
    embs_words, embs_genre = file.values()
    # embs = embs_words * (1 - w_genre) + embs_genre * w_genre
    embs = embs_words
    del file
    indices = pd.Series(data.index, index = data['title'].str.lower()).drop_duplicates()
    embs_sentence, unknown_sentence = cal_emb(process_text2(pd.Series(sentence)))
    if metric =='cosine':
        sim_vec = cosine_sim_words = metrics.pairwise.cosine_similarity(embs_sentence.reshape(1, -1), embs)

        # Get the pairwsie similarity scores of all movies with that movie
    elif metric == 'euclidean':
        sim_vec = scipy.spatial.distance.cdist(embs_sentence.reshape(1, -1), embs, metric='euclidean')
        sim_vec = 1 / np.abs(sim_vec)
    
    sim_scores = list(enumerate(sim_vec.reshape(-1,1)))
    
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    res = sorted_sim_scores[:10]

    # Get the movie indices
    res_indices = [item[0] for item in res]
    res_score = [item[1] for item in res]   
    
    final_data = data.iloc[res_indices]
    final_data.insert(0,'score',res_score)
    
    col_lst = [ 'score',  'type', 'title',  'country', 'rating',
       'listed_in', 'description']
    return final_data[col_lst]
search_recommend("hello my friend! Miss you so much",data)
    

# ----------------- Evaluate with the ratings from Netflix Prize Data
ratings = pd.read_csv('../input/selected_ratings.csv') # Distributed upon request: ryy.56@my.cityu.edu.hk
tconst = pd.read_csv("../input/title2tconst.csv")
data['tconst'] = tconst['tconst']
tconst2title = dict(zip(list(tconst['tconst']),list(tconst['title'])))
title2tconst = dict(zip(list(tconst['title']),list(tconst['tconst'])))

# data['tconst'] = data['title'].apply(lambda title: title2tconst.get(title.lower(),None))

gp = ratings.groupby(by='user')
groups = dict(ratings.groupby(by='user').groups)
users = list(groups.keys())


# save the data
with open('../data/groups_info.txt','wb') as fp:
    pickle.dump(groups,fp)
with open('../data/tconst2title.txt','wb') as fp:
    pickle.dump(tconst2title,fp)
with open('../data/title2tconst.txt','wb') as fp:
    pickle.dump(title2tconst,fp)
with open('../data/user_list.txt','wb') as fp:
    pickle.dump(users,fp)

# load the data 
with open('../data/groups_info.txt','rb') as fp:
    groups = pickle.load(fp)
with open('../data/tconst2title.txt','rb') as fp:
    tconst2title = pickle.load(fp)
with open('../data/title2tconst.txt','rb') as fp:
    title2tconst = pickle.load(fp)
with open('../data/user_list.txt','rb') as fp:
    users = pickle.load(fp)


path = 'data/user_ratings/'
n_user = len(users)
path_load = '../data/user_ratings/'
hit_lst = []
user = users[0]; i = 0
out_df = pd.DataFrame(columns=['user','No_like','like','No_hit',
                                   'No_hit_like','hit_like','avg_in_rating','avg_out_rating','time_taken(s)'])

for i in range(n_user):
    print("This is user {}.".format(i))
    user = users[i]
    start = datetime.now()
    in_df = pd.read_csv(path_load+str(user)+'.csv')
    in_df = drop_unnamed(in_df)
    
    like_df = in_df[in_df['rating'] >= 4]
    movies_like_tconst = list(like_df['movie'])
    movies_like_title = [tconst2title[m] for m in movies_like_tconst]
    No_like = len(movies_like_tconst)
    in_len = int(No_like/2)    
    movie2rating = dict(zip(in_df['movie'],in_df['rating']))
    
    
    recommend_df = pd.DataFrame(columns = data.columns)
    for movie in movies_like_title[:in_len]:
        _, _, tmp = recommend(movie,data)
        recommend_df = recommend_df.append(tmp,ignore_index=True)
    movies_recommend = list(recommend_df['tconst'].unique())
    in_df = in_df.reset_index()
    indices_user = pd.Series(in_df.index, index = in_df['movie'])
    recommend_ratings = pd.DataFrame({'movie':movies_recommend})
    recommend_ratings['rating'] = -1
    
    hit = 0
    for j in range(len(recommend_ratings)):
        try:
            recommend_ratings.loc[j,'rating'] = movie2rating[recommend_ratings.loc[j,'movie']]
            hit += 1
        except:
            pass
    recommend_res = recommend_ratings[recommend_ratings['rating']!=-1]
    out_like = recommend_res[recommend_res['rating'] >= 4]
    out_like_set = set(out_like['movie'])
    inter = set(movies_like_tconst) & out_like_set
    hit_like_title = [tconst2title[t] for t in inter]
    avg_input = np.mean(like_df.iloc[:in_len]['rating'])
    avg_output = np.mean(recommend_res['rating'])
    #'user','No_like','like','No_hit', 'No_hit_like','hit_like','avgin rating','avg_out_rating'
    end = datetime.now()
    time_taken = (end-start).total_seconds()
    out_df = out_df.append(dict(zip(out_df.columns,[user,No_like,movies_like_title,hit,
                len(inter),hit_like_title,avg_input,avg_output,time_taken])),ignore_index=True)
    
    if i % 20 == 0:
        print("{} users, {}% finished.".format(i,i/n_user*100))
    if i % 1000 == 0 & i:
        out_df.to_csv("../data/recommendation_results/genre_embeddings/evaluate_"+str(i)+'.csv')
        out_df = pd.DataFrame(columns=['user','No_like','like','No_hit',
                                   'No_hit_like','hit_like','avg_in_rating','avg_out_rating','time_taken(s)'])
        print("saved {}.".format(i))

