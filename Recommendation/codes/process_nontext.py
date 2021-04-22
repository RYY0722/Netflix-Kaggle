# # -*- coding: utf-8 -*-
# """
# Created on Sat Apr  3 16:30:41 2021

# @author: ftppr
# """

import pandas as pd
import numpy as np
from collections import Counter
data = pd.read_csv('../input/netflix_titles.csv')
bak = pd.read_csv('../input/netflix_titles.csv')
# ------- process the rating
# The ratings are groupped into 3 categories

mature = ['TV-MA','R','PG-13','TV-14','TV-PG','PG','NC-17']
general = ['NR',np.nan,"G","UR",'TV-G']
young = ['TV-Y','TV-Y7','TV-Y7-FV']
mapping = {}
mapping = {1:"Y",2:"G",3:"M"}
for item in young:
    mapping[item] = "Y"
for item in general:
    mapping[item] = "G"
for item in mature:
    mapping[item] = "M"
   
data['rating'] = data['rating'].apply(lambda item: mapping[item])


# ------- process the cast
def first_3(str_a):
    if isinstance(str_a,str):
        return str_a.lower().split(', ')[:3]
    else:
        return []
data['cast'] = data['cast'].apply(first_3)


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
bak['country'] = bak['country'].apply(split_comma) 
country_lst = []
for country in data['country']:
    country_lst += country
country_summary = dict(Counter(country_lst))
country_summary = dict(sorted(country_summary.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))
tmp = list(country_summary.values())
selected_country = dict([kv for kv in country_summary.items() if kv[1] >= 100])
selected_country_names = selected_country.keys()
print(len(selected_country))
print(selected_country_names)

selected_country_names = ['united states', 'india', 'united kingdom', 'canada', 'france', 'japan', 'spain', 'south korea', 'germany', 'mexico', 'australia', 'china', 'egypt', 'turkey', 'hong kong']
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
genre_set = []
for item in data['genre']:
    genre_set += item
# ------- process the time
# Maybe promote the newly added movies
from datetime import datetime, date
'''
which format:
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
'''

def process_time(time):
    try:
        if isinstance(time,str): 
            dt = datetime.strptime(time, '%B %d, %Y')#.date()
        else:
            dt = None        
        return dt
    except:
        print(time)
    
data['date_added'].apply(process_time)
type(data['date_added'][0])
