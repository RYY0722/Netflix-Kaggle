# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:03:21 2021

@author: ftppr
"""
import pandas as pd
import numpy as np
from utils import *
import ast
import matplotlib.pyplot as plt
import pickle
data_path = '../data/recommendation_results/onehot_plus_embeddings/evaluate_'
df = pd.read_csv(data_path+str(1000)+'.csv')



for i in range(2,297):
    df = df.append(pd.read_csv(data_path+str(i*1000)+'.csv'))
df.columns
sub = df[['user', 'No_like', 'No_hit', 'No_hit_like','avg_in_rating',
          'avg_out_rating', 'time_taken(s)']]
# Convert back to dict

# load the data 
with open('../data/total_watched.txt','rb') as fp:
    total_watched = pickle.load(fp)
    
list(total_watched.keys())[:10]
df['total_watch'] = list(total_watched.values())[:len(df)]

df['out_freq']=df['out_freq'].apply(lambda x: ast.literal_eval(x))
for i in range(1,6):
    col = "rate_"+str(i)
    df[col] = df['out_freq'].apply(lambda row: row.get(i,0))
sub['total_watch'] = df['total_watch']
sub.columns
sub = df[['user', 'total_watch', 'No_like', 'No_hit', 'No_hit_like', 'avg_in_rating',
       'avg_out_rating',  'rate_1', 'rate_2',
       'rate_3', 'rate_4','rate_5']]

# Calculate %
sub['hit%'] = sub['No_hit'] / sub['total_watch'] * 100
sub['hit like%'] = sub['No_hit_like'] / sub['total_watch'] * 100
sub['rate 5%'] = sub['rate_5'] / sub['No_hit'] * 100
sub['rate 4%'] = sub['rate_4'] / sub['No_hit'] * 100
sub['rate 3%'] = sub['rate_3'] / sub['No_hit'] * 100
sub['rate 2%'] = sub['rate_2'] / sub['No_hit'] * 100
sub['rate 1%'] = sub['rate_1'] / sub['No_hit'] * 100

bak = sub

# Convert NA to 0
for col in sub.columns:
    sub[col] = sub[col].fillna(0)

sub['rate 5%'].mean()

sub = sub.sort_values(by=['total_watch'])
seps = [0, 20, 50, 100]
seps_idx = [sum(sub['total_watch']<sep) for sep in seps]

seps_idx = seps_idx + [len(sub)]

sub_dict = {}
tmp = 0
for i in range(len(seps_idx) - 1):
    sub_dict[i] = sub.iloc[seps_idx[i]:seps_idx[i+1]]
    tmp += len(sub_dict[i])
if tmp == len(sub):
    print("No missing value!")

stats = pd.DataFrame(columns = ['total_watch', 'No_like', 'No_hit', 'No_hit_like',
       'avg_in_rating', 'avg_out_rating', 'rate_1', 'rate_2', 'rate_3',
       'rate_4', 'rate_5', 'hit%', 'hit like%', 'rate 5%', 'rate 4%', 'rate 3%',
       'rate 2%', 'rate 1%'])
for i,tmp in sub_dict.items():
    stats.loc[i] = tmp.drop(columns=['user']).mean()

# ------------- plot the data



plt.plot( stats['No_like'],stats['avg_in_rating'], label = 'in rating',marker='o', markersize=8,  linewidth=2)
plt.plot( stats['No_like'], stats['avg_out_rating'],label = 'out rating', marker='o', markersize=8, linewidth=2)
plt.legend()#"input rating",'output rating'
plt.xlabel("Number of movie like")
plt.ylabel("rating")
plt.show()




# plot hit rate  markerfacecolor='blue',color='skyblue',
plt.close()
plt.plot( stats['No_like'],stats['hit%'], label = 'hit (%)',marker='o', markersize=8,  linewidth=2)
plt.plot( stats['No_like'], stats['hit like%'],label = 'hit like(%)', marker='o', markersize=8, linewidth=2)
plt.legend()#"input rating",'output rating'
plt.xlabel("Number of movie like")
plt.ylabel("hit rate (%)")
plt.show()


plt.close()
plt.plot( [4.62,18.93,38.92,68.14],[26.64,62.48,60.19,55.25], label = 'hit (%)',marker='o', markersize=8,  linewidth=2)
plt.plot( [4.62,18.93,38.92,68.14], [5.80,15.88,23.02,28.01],label = 'hit like(%)', marker='o', markersize=8, linewidth=2)
plt.legend()#"input rating",'output rating'
plt.xlabel("Number of movie like")
# plt.ylabel("hit rate (%)")
plt.show()



# plot rating rates
plt.close()
plt.plot( stats['No_like'],stats['rate 1%'], label = 'rate 1 (%)',marker='o', markersize=8,  linewidth=2)
plt.plot( stats['No_like'],stats['rate 2%'], label = 'rate 2 (%)',marker='o', markersize=8,  linewidth=2)
plt.plot( stats['No_like'],stats['rate 3%'], label = 'rate 3 (%)',marker='o', markersize=8,  linewidth=2)
plt.plot( stats['No_like'],stats['rate 4%'], label = 'rate 4 (%)',marker='o', markersize=8,  linewidth=2)
plt.plot( stats['No_like'],stats['rate 5%'], label = 'rate 5 (%)',marker='o', markersize=8,  linewidth=2)
plt.legend(loc='best')#"input rating",'output rating'
plt.xlabel("Number of movie like")
plt.ylabel("rating percentage (%)")
plt.show()

# show legend
plt.legend()

# show graph
plt.show()

