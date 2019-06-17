#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import interp

import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='monospace', style='normal', size=10)

df = pd.read_csv('dataset_experimental_working.csv', sep=';')
df_nonan = df.dropna()


matches = df[df.loc[:, 'match'] == 1]
female_likes = df[(df.loc[:, 'gender'] == 'female') & (df.loc[:, 'dec'] == 1)]
female_likes_samerace = female_likes[female_likes.loc[:, 'race'] == female_likes.loc[:, 'race_o']]
female_likes_diffrace = female_likes[female_likes.loc[:, 'race'] != female_likes.loc[:, 'race_o']]
female_likes_samerace_imp = [len(female_likes_samerace[female_likes_samerace.loc[:, 'imp_same_race'] == i]) for i in range(1, 11)]
female_likes_diffrace_imp = [len(female_likes_diffrace[female_likes_diffrace.loc[:, 'imp_same_race'] == i]) for i in range(1, 11)]
female_likes_samerace_samerel_imp = [len(female_likes_samerace[female_likes_samerace.loc[:, 'imp_same_religion'] == i]) for i in range(1, 11)]
female_likes_diffrace_samerel_imp = [len(female_likes_diffrace[female_likes_diffrace.loc[:, 'imp_same_religion'] == i]) for i in range(1, 11)]


male_likes = df[(df.loc[:, 'gender'] == 'male') & (df.loc[:, 'dec'] == 1)]
male_likes_samerace = male_likes[male_likes.loc[:, 'race'] == male_likes.loc[:, 'race_o']]
male_likes_diffrace = male_likes[male_likes.loc[:, 'race'] != male_likes.loc[:, 'race_o']]
male_likes_samerace_imp = [len(male_likes_samerace[male_likes_samerace.loc[:, 'imp_same_race'] == i]) for i in range(1, 11)]
male_likes_diffrace_imp = [len(male_likes_diffrace[male_likes_diffrace.loc[:, 'imp_same_race'] == i]) for i in range(1, 11)]
male_likes_samerace_samerel_imp = [len(male_likes_samerace[male_likes_samerace.loc[:, 'imp_same_religion'] == i]) for i in range(1, 11)]
male_likes_diffrace_samerel_imp = [len(male_likes_diffrace[male_likes_diffrace.loc[:, 'imp_same_religion'] == i]) for i in range(1, 11)]


matches_samerace = matches[matches.loc[:, 'race'] == matches.loc[:, 'race_o']]
matches_diffrace = matches[matches.loc[:, 'race'] != matches.loc[:, 'race_o']]
matches_samerace_samerace_imp = [len(matches_samerace[matches_samerace.loc[:, 'imp_same_race'] == i]) for i in range(1, 11)]
matches_diffrace_samerace_imp = [len(matches_diffrace[matches_diffrace.loc[:, 'imp_same_race'] == i]) for i in range(1, 11)]
matches_samerace_samerel_imp = [len(matches_samerace[matches_samerace.loc[:, 'imp_same_religion'] == i]) for i in range(1, 11)]
matches_diffrace_samerel_imp = [len(matches_diffrace[matches_diffrace.loc[:, 'imp_same_religion'] == i]) for i in range(1, 11)]


avg_imp_same_race = np.average(df_nonan.loc[:, 'imp_same_race'])
avg_imp_same_religion = np.average(df_nonan.loc[:, 'imp_same_religion'])
avg_imp_attr = np.average(interp(df_nonan.loc[:, 'imp_attr'], [1, 100], [1, 10]))
avg_imp_sinc = np.average(interp(df_nonan.loc[:, 'imp_sinc'], [1, 100], [1, 10]))
avg_imp_intel = np.average(interp(df_nonan.loc[:, 'imp_intel'], [1, 100], [1, 10]))
avg_imp_fun = np.average(interp(df_nonan.loc[:, 'imp_fun'], [1, 100], [1, 10]))
avg_imp_amb = np.average(interp(df_nonan.loc[:, 'imp_amb'], [1, 100], [1, 10]))
avg_imp_shared_interests = np.average(interp(df_nonan.loc[:, 'imp_shared_interests'], [1, 100], [1, 10]))
avg_imp = [avg_imp_amb, avg_imp_shared_interests, avg_imp_fun, avg_imp_sinc, avg_imp_intel, avg_imp_attr, avg_imp_same_religion, avg_imp_same_race][::-1]
avg_imp_labels = ["imp_amb", "imp_shared_interests", "imp_fun", "imp_sinc", "imp_intel", "imp_attr", "imp_same_religion", "imp_same_race"][::-1]


df[df.loc[:, 'race'] == 'Asian/Pacific Islander/Asian-American']    # 1982
df[df.loc[:, 'race'] == 'European/Caucasian-American']              # 4727
df[df.loc[:, 'race'] == 'Other']                                    # 522
df[df.loc[:, 'race'] == 'Latino/Hispanic American']                 # 664
df[df.loc[:, 'race'] == 'Black/African American']                   # 420



plt.title("Mean values for importance attributes")
plt.barh(avg_imp_labels, avg_imp, height=0.5, color='#65c7eb')
plt.ylim([-1, 8])
filename = 'mean_importances-%s.pdf' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
plt.show()


