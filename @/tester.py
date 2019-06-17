# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:04:20 2018

@author: ae0670
"""
import numpy as np
import time
#import unique_combinations as unicomb
#sys.path.append('/metrics')
#from metrics.simpledtw import dtw
from algorithms.exhaustive import Exhaustive
from algorithms.greedy import Greedy
from collections import Counter
import textdistance

probsame_cand = np.array([0,0,1,0,0,1,1,0,1,1])
scores_cand = np.array([0.40,0.8,0.30,0.6,0.51,0.49,0.90,0.7,0.20,0.10])
pref = 0
b = 0.5

# sort both arrays in ascending order of scores
s = sorted(zip(scores_cand, probsame_cand))
scores_cand, probsame_cand = map(np.array, zip(*s))

print(probsame_cand)
print(scores_cand)

# create recommendation sets
split_index = np.argmax(scores_cand >= b)
scores_rec = scores_cand[split_index:]
probsame_rec = probsame_cand[split_index:]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

'''
def combinations(subset_size):
    comb_iterator = unicomb.combinations(imp_cand, subset_size)
    combinations = np.stack(comb_iterator)
    means = np.asarray(list(map(np.mean, combinations)))
    return combinations, means
'''

start = time.time()

'''
data = Data(scores_cand, imp_cand)
index_map = data.index_map
imp_map = data.index_map
'''
'''
#_, means = combinations(15)
#print (means)
subsets = itertools.combinations(imp_cand, 15)
subsets = np.asarray(list(dict.fromkeys(subsets))) # remove duplicates
subset_means = np.asarray(list(map(np.mean, subsets)))
print (subset_means)
'''
'''
e = Epsilon(pref, probsame_cand, probsame_rec)
print (e.m_star())
print (e.score())
e.visualize([0.0, 0.5, 1.0])
'''
'''
rel = Relevance(scores_cand, b)
print('relevance of rec.set: {}'.format(rel.score(scores_rec)))
'''
#scores_rest = scores_cand[:split_index]
#probsame_rest = probsame_cand[:split_index]
#print(scores_rest)
#print(probsame_rest)


start = time.time()
alg_exhaustive = Exhaustive(scores_cand, probsame_cand, b)
result_exhaustive = np.sort(alg_exhaustive.run(pref=1, alpha=0))
print('optimal subset: {}'.format(result_exhaustive))
print ('\nruntime: {}'.format(time.time()-start)) 
start = time.time()
alg_greedy = Greedy(scores_cand, probsame_cand, b)
result_greedy = np.sort(alg_greedy.run(pref=1, alpha=0))
print('greedily optimal subset: {}'.format(result_greedy))
print ('\nruntime: {}'.format(time.time()-start))

r1 = Counter(list(result_exhaustive))
r2 = Counter(list(result_greedy))
print(r2.items)
print (np.all(result_exhaustive == result_greedy))
diff = r1-r2
print ('set difference: {}'.format(diff.elements()))
result_exhaustive=[0.2, 0.2, 0.8]
result_greedy=[0.2, 0.2]
s1 = list(map(str,np.sort(result_exhaustive)))
s2 = list(map(str,np.sort(result_greedy)))
print (s1)
print (s2)
#s1 = ['t','e','s','t']
#s2 = ['t','e','t']
#print('modified optimal subset: {}'.format(result_exhaustive))
edit_distance = levenshteinDistance(s1, s2)
#print ('kendall\'s tau: {}'.format(stats.kendalltau(np.sort(result_exhaustive), np.sort(result_greedy))))
#matches, dtw_cost, mapping_1, mapping_2, matrix = dtw(s1, s2)
dice = textdistance.sorensen_dice(s1,s2)
jaccard = textdistance.jaccard(s1,s2)
print ('edit distance: {}'.format(edit_distance))
#print ('dtw distance: {}'.format(dtw_cost))
print ('dice: {}'.format(dice))
print ('jaccard: {}'.format(jaccard))

c1 = Counter(result_exhaustive)
c2 = Counter(result_greedy)
diff=c1-c2
print ('C1: {}'.format(list(diff.items())))
print ('C2: {}'.format(c2))

print ('\nCombined counts:')
print (c1 + c2)

zipped = zip(scores_cand, probsame_cand)
