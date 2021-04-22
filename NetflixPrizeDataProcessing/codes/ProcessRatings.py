# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:25:17 2021

process netflix prize

@author: ftppr
"""

from datetime import datetime
import os
import random
import matplotlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from scipy import sparse
from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
#from sklearn.metrics.pariwise import cosine_similarity

start = datetime.now()
if not os.path.isfile('data.csv'):
    #read all txt file and store them in one big file
    data = open('data.csv', mode='w')
    
    row = list()
    files = ['../input/netflix-prize-data/combined_data_1.txt', '../input/netflix-prize-data/combined_data_2.txt',
            '../input/netflix-prize-data/combined_data_3.txt', '../input/netflix-prize-data/combined_data_4.txt']
    for file in files:
        print('reading ratings from {}...'.format(file))
        with open(file) as f:
            for line in f:
                del row[:]
                line = line.strip()
                if line.endswith(':'):
                    #all are rating
                    movid_id = line.replace(':', '')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(0, movid_id)
                    data.write(','.join(row))
                    data.write('\n')
        print('Done.\n')
    data.close()
print('time taken:', datetime.now() - start)

print('creating the dataframe from data.csv file..')
df = pd.read_csv('data.csv', sep=',', names=['movie','user','rating','date'])

# df.date = pd.to_datetime(df.date)
print('Done.\n')

#arranging the rating according to time
print('sorting the dataframe by date..')
df.sort_values(by='date', inplace=True)
print('sorting done.')


df.head()
df.describe()
df.describe()['rating']
dup = df.duplicated(['movie','user','rating'])
dups = sum(dup) #considering by column
print('there are {} duplicate rating entries in the data.....'.format(dups))
print('number of NaN values in our dataset:', sum(df.isnull().any()))


# read in description of movies
movie_df = pd.read_csv('../input/netflix-prize-data/movie_titles.csv',error_bad_lines=False, encoding='latin1')
movie_df['title'] = movie_df['title'].str.lower()
movie_df = movie_df.drop_duplicates(subset=['title'])
movie_rating_set = set(movie_df['title'].unique())

data = pd.read_csv("../input/titles_cleaned.csv")
print("read in cleaned titles succesfully")
data['title'] = data['title'].str.lower()
movie_data = set(data['title'])
intersect = movie_data & movie_rating_set
print("There are {} movies in common.".format(len(intersect)))  # 635

title_lst = list(movie_df['title'])
indices = [i for i in range(len(title_lst)) if title_lst[i] in intersect]
if len(indices) != len(intersect):
    print("Len of indices is {} while intersect is {}".format(len(indices),len(intersect)))

data['tconst'] = ''
tconst_set = []
for idx in indices:
    rec = movie_df.iloc[idx]
    d = data['title'] == rec['title']
    if sum(d) > 1:
        print("there are {} movies with name {}".format(sum(d),rec['title']))
    if sum(d) < 1:
        print("there is no movie named "+rec['title'])
    d_idx = list(d).index(True)
    data.loc[d_idx,'tconst'] = rec['tconst']
    tconst_set.append(rec['tconst'])
data_intersect = data[data['tconst']!='']
if len(data_intersect) != len(indices):
    print("number of selected dataframe columns is wrong")
try:
    data_intersect = data_intersect.drop(columns=['Unnamed: 0'])
except:
    print("no columns named Unnamed: 0")
    pass
# data_intersect.to_csv('../data/title2tconst.csv')
print("dataframe of title2tconst is created")
# selected_ratings = df.apply(lambda row: row if row['movie'] in tconst_set else None,axis=1)
len_df = len(df)
print("Done!")
# split df because it is too large....





sub_df = {}
ind = list(range(len_df))
num_slices = 2000
per = int(np.ceil(len_df/num_slices))
for i in range(num_slices):
    indices = ind[i*per : (i+1)*per]
    sub_df[i] = df.iloc[indices]
    print("No.{} dataframe created".format(i))

tmp = 0
for i in range(len(sub_df)):
    tmp += len(sub_df[i])
if tmp != len_df:
    print("Difference is {}".format(tmp - len_df))
else:
    print("sub df and df have the same number of entries!")


    
# process the ratings
def process_ratings(df,tconst_set):
    start = datetime.now()
    selected_ratings = pd.DataFrame(columns=['movie', 'user', 'rating', 'date'])
    for i in range(len(df)):
        rec = df.iloc[i]
        if rec['movie'] in tconst_set:
            selected_ratings = selected_ratings.append(rec)
        # if i % 10000 == 0:
        #     print("finished {}".format(i/len(df)))
    print("There are {} ratings left in the current batch".format(len(selected_ratings)))
    print('time taken for this batch:', datetime.now() - start)
    return selected_ratings

# selected_ratings_sub = {}
total_selected_ratings = 0
for i in range(1886,num_slices):
    selected_ratings_sub = process_ratings(sub_df[i],tconst_set)
    print("Done {}".format((i+1)/num_slices))
    selected_ratings_sub.to_csv('../data/processed_ratings/slice_'+str(i)+'.csv')
    total_selected_ratings += len(selected_ratings_sub)
print("Finished!!\n\nThere are {} ratings left".format(total_selected_ratings))



base_path = "../data/processed_ratings/slice_"
ratings = pd.read_csv(base_path+'0.csv')
for i in range(1,10):
    ratings = ratings.append(pd.read_csv(base_path+str(i)+'.csv'))
try:
    ratings = ratings.drop(columns=['Unnamed: 0'])
    print('Dropped unnamed 0')
except:
    pass
finally:
    print("Columns are" )
    print(ratings.columns)
users = set(ratings['user'])
print("Every user has around {} records".format(len(ratings)/len(users)))

# Merge all csv files

from os import listdir
from os.path import isfile, join
path = "../data/processed_ratings/"
files = [f for f in listdir(path) if isfile(join(path, f))]
all_ratings = pd.DataFrame(columns=['movie', 'user', 'rating', 'date'])
for i,f in enumerate(files):
    try:
        all_ratings = all_ratings.append(pd.read_csv(path+f))
        if i % 100 == 0:
            print("{}% finished".format(i/2000*100))
    except:
        pass

print("There are {} ratings.".format(len(all_ratings)))
# There are 12429975 ratings.
all_ratings = all_ratings.sort_values(by="user")

try:
    all_ratings = all_ratings.drop(columns="Unnamed: 0")
except:
    pass
finally:
    print("Columns are: ")
    print(all_ratings.columns)

all_ratings.to_csv('../data/selected_ratings.csv')
a = all_ratings.groupby('user')

all_ratings = pd.read_csv('../data/selected_ratings.csv')
# np.save('../data/groups.npy', a.groups) 
groups = np.load('../data/groups.npy',allow_pickle='TRUE').item()


# import json
# with open('../data/groups.json', 'w') as f:
#     json.dump(a.groups, f)

# elsewhere...

# with open('my_dict.json') as f:
#     my_dict = json.load(f)



users = set(all_ratings['user'])
print("There are {} users.".format(len(users)))
print("Every user has around {} records".format(len(all_ratings)/len(users)))
# There are 462732 users.
# Every user has around 26.862146987889318 records


# Sample program for suprise
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.reader import Reader
from surprise.model_selection import KFold
from surprise import accuracy
# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
ratings = all_ratings.iloc[0:10**6]
data2 = pd.DataFrame({'userID':ratings['user'],'itemID':ratings['movie'],
                      'rating':ratings["rating"]})

data2['rating'].describe()


data = Dataset.load_from_df(data2, Reader( line_format=u'user item rating',rating_scale=(1,5)))
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# define a cross-validation iterator
kf = KFold(n_splits=5)
time1 = datetime.now()
for trainset, testset in kf.split(data):
    time2 = datetime.now()
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    print("hi:{}".format(time2 -time1))
    time1 = time2














