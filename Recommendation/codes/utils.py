# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 09:37:24 2021

@author: ftppr
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import pandas as pd

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

def drop_unnamed(df):
    drop_columns = [col for col in df.columns if col[:7] == "Unnamed"]
    try: 
        return df.drop(columns = drop_columns)
    except:
        return df
    

def cal_emb(data,embeddings=None):    
    from gensim.models import word2vec
    # sentences = word2vec.Text8Corpus('../input/text8/text8')
    # training
    # model = word2vec.Word2Vec(sentences, vector_size=100)#, window=5, min_count=1, workers=4)
    
    # model.save('text8.model')
    model = word2vec.Word2Vec.load("../model/text8.model")
    # model.save('text8-635-100.model')
    # model = word2vec.Word2Vec.load("text8-635-100.model")
    embeddings = model.wv
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

def scale(data):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled

def recommend_1(title, data, metric='cosine',w_genre = 0.3):
    # files = np.load('../data/dist.npz')
    files = np.load('../data/dist-sub.npz')
    cosine_sim_genre = files['arr_0']
    cosine_sim_words = files['arr_1']
    euclidean_sim_genre = files['arr_2']
    euclidean_sim_words = files['arr_3']
    del files # save memory
    indices = pd.Series(data.index, index = data['title'].str.lower()).drop_duplicates()
    idx = indices[title.lower()]
    if metric =='cosine':
        sim_genre_vec = cosine_sim_genre[idx]
        sim_words_vec = cosine_sim_words[idx]
        # Get the pairwsie similarity scores of all movies with that movie
        
        
    elif metric == 'euclidean':
        sim_genre_vec = euclidean_sim_genre[idx]
        sim_words_vec = euclidean_sim_words[idx]
        
    sim_scores_words = list(enumerate(sim_words_vec))
    sim_scores_genre = list(enumerate(sim_genre_vec))
        
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
                         'recommend by genre':data['title'].iloc[genre_base_indices],
                         'score by words':words_base_score})
    res3 = pd.DataFrame({'recommend by words':data['title'].iloc[words_base_indices]})
    res3.index = range(len(res3))
    res1.index = range(len(res1))
    res1['recommend by words'] = res3['recommend by words']
    # res1 = res1.iloc[1:]
    w_genre = w_genre
    final_score =  sim_words_vec * (1 - w_genre) + sim_genre_vec * w_genre
    final_score = list(enumerate(final_score))
    sorted_final_score = sorted(final_score, key=lambda x: x[1], reverse=True)
    final = sorted_final_score[:11]
    final = [item for item in final if int(item[0]) != idx]
    final_indices = [i[0] for i in final]
    final_selected_score = [i[1] for i in final]
    
    res2 = pd.DataFrame({'title':title,'score by genre':sim_genre_vec[final_indices],
                         'score by words':sim_words_vec[final_indices],
                         'overall score':final_selected_score,
                         'overall recommendation':data['title'].iloc[final_indices]})
    # res2 = res2.iloc[1:]
    
    final_data = data.iloc[final_indices]
    return res1,res2,final_data


def first_3(str_a):
    if isinstance(str_a,str):
        return str_a.lower().split(', ')[:3]
    else:
        return []
    
    
def split_comma(str_a):
    if isinstance(str_a,str):
        return str_a.lower().split(', ')[:3]
    else:
        return []
