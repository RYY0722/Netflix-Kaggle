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
data = pd.read_csv('netflix_titles.csv')
bak = data

from gensim.models import word2vec
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim.downloader
import numpy as np
import string

# Use pretrained model
# glove_vectors = gensim.downloader.load('glove-twitter-200')

# train the model with data incorporating description
# --------------- Get the embeddings --------------

def process_text2(data):
    data = data.str.lower()
    data = data.apply(lambda row: word_tokenize(row))
    data = data.apply(lambda row: [word for word in row if word.isalpha() and not word in stopwords.words('english')])
    useless_pos_tags = ['CD', 'MD', 'RB', 'RBS ', 'RBR', 'WRB', 'WP']
    data = data.apply(lambda row: [word for (word, tag) in nltk.pos_tag(row) if tag not in useless_pos_tags])
    data = data.apply(lambda row: [word for word in row if word not in list(string.punctuation) + list("@#$%^&*()_+=-……￥·【】[]|\{}") ])
    lemmatizer=WordNetLemmatizer()
    data = data.apply(lambda row: [lemmatizer.lemmatize(word) for word in row])
    return data
data['words'] = process_text2(data['title'] + ' ' + data['description'])
data['genre'] = process_text2(data['listed_in'])
more_sentences = []
words_set = set()
for i in range(len(data['words'])):
    tmp = data['words'].iloc[i]
    more_sentences.append(tmp)
    words_set.update(tmp)
str_total = ''
for lst in more_sentences:
    str_add = ' '.join(lst)
    str_total += str_add
file = 'text8/text8'
with open(file, 'w+',encoding='utf-8') as f:
    f.write(str_total)
# model.build_vocab(more_sentences)
# model.train(more_sentences,total_examples=len(more_sentences), total_words=len(words_set), epochs=10)

from gensim.models import word2vec
sentences = word2vec.Text8Corpus('text8/text8')
# training
model = word2vec.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

model.save('text8.model')
model = word2vec.Word2Vec.load("text8.model")
glove_vectors = model.wv

def cal_emb(data):    
    embs = np.zeros([len(data),glove_vectors.vector_size])
    unknown = []
    for i in range(len(data)):
        row = (data.iloc[i])#.tolist()[0]
        cnt = 0
        for word in row:
            try:
                embs[i] += glove_vectors[word]            
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
embs_genre, unknown_genre = cal_emb(data['genre'])
import scipy.spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing

def scale(data):
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled
embs_words = scale(embs_words)
embs_genre = scale(embs_genre)

cosine_sim_genre = 1 - scipy.spatial.distance.cdist(embs_genre, embs_genre, metric='cosine')
cosine_sim_words = 1 - scipy.spatial.distance.cdist(embs_words, embs_words, metric='cosine')

#index mapping of the orginal title
# data1 = pd.read_csv('netflix_titles.csv')
indices = pd.Series(bak.index, index = bak['title']).drop_duplicates()
#print(indices)
euclidean_words = scipy.spatial.distance.cdist(embs_words, embs_words, metric='euclidean')
euclidean_genre = scipy.spatial.distance.cdist(embs_genre, embs_genre, metric='euclidean')
#jaccard = scipy.spatial.distance.cdist(embs, embs,  metric='jaccard')

np.savez('dist.npz', cosine_sim_genre,cosine_sim_words,euclidean_genre,euclidean_words)  


def recommend(title, metric='cosine'):
    files = np.load('dist.npz')
    cosine_sim_genre = files['arr_0']
    cosine_sim_words = files['arr_1']
    euclidean_sim_genre = files['arr_2']
    euclidean_sim_words = files['arr_3']
    del files # save memory
    idx = indices[title]
    if metric =='cosine':
        
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores_words = list(enumerate(cosine_sim_words[idx]))
        sim_scores_genre = list(enumerate(cosine_sim_genre[idx]))
        
    elif metric == 'euclidean':
        sim_scores_words = list(enumerate(euclidean_sim_words[idx]))
        sim_scores_genre = list(enumerate(euclidean_sim_genre[idx]))
        
    # Sort the movies based on the similarity scores
    sorted_sim_scores_words = sorted(sim_scores_words, key=lambda x: x[1], reverse=True)
    sorted_sim_scores_genre = sorted(sim_scores_genre, key=lambda x: x[1], reverse=True)
    
    genre_base = sorted_sim_scores_genre[:11]
    words_base = sorted_sim_scores_words[:11]

    # Get the movie indices
    genre_base_indices = [i[0] for i in genre_base]
    words_base_indices = [i[0] for i in words_base]
    genre_base_score = [i[1] for i in genre_base]
    words_base_score = [i[1] for i in words_base]
    res1 = pd.DataFrame({'title':title,'score by genre':genre_base_score,
                         'recommend by genre':bak['title'].iloc[genre_base_indices],
                         'score by words':words_base_score})
    res3 = pd.DataFrame({'recommend by words':bak['title'].iloc[words_base_indices]})
    res3.index = range(len(res3))
    res1.index = range(len(res1))
    res1['recommend by words'] = res3['recommend by words']
    # res1 = res1.iloc[1:]
    w_genre = 0.5
    final_score =  cosine_sim_words[idx] * (1 - w_genre) + cosine_sim_genre[idx] * w_genre
    final_score = list(enumerate(final_score))
    sorted_final_score = sorted(final_score, key=lambda x: x[1], reverse=True)
    final = sorted_final_score[:11]
    final_indices = [i[0] for i in final]
    final_selected_score = [i[1] for i in final]
    
    res2 = pd.DataFrame({'title':title,'score by genre':cosine_sim_genre[idx][final_indices],
                         'score by words':cosine_sim_words[idx][final_indices],
                         'overall score':final_selected_score,
                         'overall recommendation':bak['title'].iloc[final_indices]})
    # res2 = res2.iloc[1:]
    return res1,res2

title = 'Peaky Blinders'
metric='cosine'
res1, res2 = recommend(title, metric)









