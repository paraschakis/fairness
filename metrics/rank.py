# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 23:56:30 2018

@author: ae0670
"""

from collections import Counter

def intersection(s1, s2):
    s1 = [round(x, 2) for x in s1]
    s2 = [round(x, 2) for x in s2]
    c1 = Counter(s1)
    c2 = Counter(s2)
    return sum((c1 & c2).values())    

def dice_similarity(s1, s2):
    return 2*intersection(s1,s2)/(len(s1)+len(s2))

def jaccard_similarity(s1, s2):
    return intersection(s1,s2)/(len(s1)+len(s2)-intersection(s1,s2))

def f1(s1, s2): # s1: reference ranking, s2: evaluated ranking
    recall = intersection(s1,s2)/len(s1)
    precision = intersection(s1,s2)/len(s2)
    if 0.0 in (recall, precision):
        return 0.0
    else:
        return 2*recall*precision/(recall+precision)