# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:02:25 2019

@author: ae0670
"""
import numpy as np
import pandas as pd
from itertools import product
import load_data

# load data and create new column
OUTPUT_FILE = '../data/processed/speed_dating_religion.csv'
df = load_data.raw()
df.insert(loc=9, column='same_religion', value=np.nan)

# build 'religion by race' table from:
# https://www.pewforum.org/religious-landscape-study/racial-and-ethnic-composition/

columns = ['Buddhist', 'Catholic', 'Evangelical Protestant', 'Hindu',
           'Historically Black Protestant', 'Jehovah Witness', 'Jewish',
           'Mainline Protestant', 'Mormon', 'Muslim', 'Orthodox Christian',
           'Other Christian', 'Other Faiths', 'Other World Religions',
           'Unaffiliated', 'Dont know']

rows = ['European/Caucasian-American', 'Black/African American',
        'Asian/Pacific Islander/Asian-American', 'Latino/Hispanic American',
        'Other']

# percentages of belonging to religion per race
data = np.array([[1, 19, 29, 1, 1, 1, 3, 19, 2, 1, 1, 1, 2, 1, 24, 1],  # white
                 [1, 5, 14, 1, 53, 2, 1, 4, 1, 2, 1, 1, 1, 1, 18, 1],   # black
                 [6, 17, 11, 16, 1, 1, 1, 5, 1, 6, 1, 1, 1, 3, 31, 1],  # asian
                 [1, 48, 19, 1, 1, 2, 1, 5, 1, 1, 1, 1, 1, 1, 20, 1],   # latino
                 [2, 10, 34, 1, 1, 1, 1, 13, 2, 1, 1, 1, 4, 1, 27, 1]]) # other

table = pd.DataFrame(data, rows, columns)

# estimate the probability of two races sharing the same religion
def prob_same_rel(pair):
    # get probability vectors
    race1 = table.loc[pair[0], :] / 100
    race2 = table.loc[pair[1], :] / 100
    total_prob = 0
    for i in range(len(table.columns)):
        total_prob += race1[i] * race2[i]
    return total_prob

# compute probabilities of same religion for all records and save the file
for permutation in product(rows, repeat=2): # cartesian product
    df['same_religion'] = np.where(
            (df['race'] == permutation[0]) & (df['race_o'] == permutation[1]),
            prob_same_rel(permutation), df['same_religion'])
df.to_csv(OUTPUT_FILE, index=False)