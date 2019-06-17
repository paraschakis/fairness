# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:02:39 2019

@author: ae0670
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('..'));
from algorithms.knapsack import Knapsack
from algorithms.tabu import Tabu
from metrics.epsilon import Epsilon
import time

#scores = np.array([0.02, 0.04, 0.12, 0.14, 0.17, 0.33, 0.36, 0.38, 0.55, 0.58])
#probsame = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])

#scores = np.array([0.4,0.3,0.2,0.1,0.01,-0.01,-0.1,-0.2,-0.3,-0.4])
#scores = np.array([0.9,0.8,0.7,0.6,0.51,0.89,0.70,0.60,0.80,0.80])
scores = np.array([0.9,0.8,0.7,0.6,0.51,0.49,0.40,0.30,0.20,0.10])
probsame = np.array([0,0,1,0,0,1,1,0,1,1])
#probsame = np.array([0.22,0.19,0.72,0.82,0.93,0.14,0.25,0.15,0.13,0.96])

#scores = np.array([0.1,0.71,0.83])
#probsame = np.array([1,0,0])

#scores = np.array([0.1,0.2,0.3,0.4,0.6])
#probsame = np.array([1,1,1,0,0])

#scores = np.array([0.1, 0.25, 0.37, 0.48, 0.8, 0.88, 0.92])
#probsame = np.array([0.96, 0.68, 0.83, 0.67, 0.81, 0.27, 0.94])

#scores = np.array([0.11,0.21,0.29,0.65,0.78,0.85,0.34,0.60,0.9,0.70,0.63,0.23,0.53,0.45,0.74,0.70,0.63,0.25,0.53,0.05])
#probsame = np.array([0.42,0.49,0.19,0.22,0.65,0.98,0.35,0.93,0.05,0.21,0.72,0.07,0.86,0.11,0.53,0.79,0.23,0.57,0.13,0.84])





#scores = np.array([0.1,0.4,0.5,0.7])
#probsame = np.array([0.1,0.3,0.4,0.5])

#scores = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#scores = np.array([-0.1,0.4,0.3,-0.4,0.5,-0.2,0.3,0.1,-0.4])
#probsame = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

#scores = np.array([0.31, 0.32, 0.97])
#probsame = np.array([0.36, 0.59, 0.4])
#pref = 0.19567502981497398

pref = 0
#pref = 0.233
b = 0.5


#probsame = np.array([0,0,0,0,0,0,0,0,0,0])
#probsame_rec = np.array([0, 0])
#
#e = Epsilon(pref, probsame, probsame_rec)
#print(e.score())



#start = time.time()
#alg_tabu = Tabu(pref, scores, probsame, b, rel_func=lambda w,s: w*s)
#result_tabu = alg_tabu.run(alpha=0.0)
#print('tabu solution: {}'.format(result_tabu))
#print('runtime: {}'.format(time.time()-start))
'''

probsame_rec = np.array([0.4190654849476694, 0.33702980640595503, 0.37345812787239663])
e = Epsilon(pref, probsame, probsame_rec)
print('epsilon: {}'.format(e.score()))
_, _, subset_min, _ = e.delta_bounds()
print('subset_min: {}'.format(subset_min))
'''

'''
scores = np.array([0., 0.03, 0.18, 0.23, 0.34, 0.44, 0.51, 0.53, 0.86, 0.97])
probsame = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
w1 = np.array([0.3, 0.74, 0.2])
w2 = np.array([1, 0, 0])

v1 = np.array([0.44, 0.51, 0.53, 0.86, 0.97])
v2 = np.array([0.51, 0.86, 0.97])
'''
'''
start = time.time()
alg_exhaustive = Exhaustive(scores, probsame, b, rel_func=lambda w,s: s)
result_exhaustive, _ = alg_exhaustive.run(pref=pref, alpha=0)
print('exhaustive solution: {}'.format(result_exhaustive))
print('runtime: {}'.format(time.time()-start))
#print(sum(result_exhaustive-b))
'''

start = time.time()
#alg = Knapsack(pref, scores, probsame, b, prec=2, 
#                        return_ids=True, rel_func=lambda w,s: w*s)
alg = Tabu(pref, scores, probsame, b, return_ids=True, rel_func=lambda w,s: w*s)
#result_tabu = alg.run(pref=pref, alpha=0)
#best_subset = alg_knapsack.run(pref=pref, alpha=0, capacity_interval=(2,3))
ids = alg.run(pref=pref, alpha=0)
#print('knapsack solution: {0}'.format(result_knapsack))
#print('tabu solution: {}'.format(result_tabu))
print('solution scores: {0}'.format(scores[ids]))
print('solution probsame: {0}'.format(probsame[ids]))
print('runtime: {}'.format(time.time()-start))
#print(sum(best_subset-b))
#e = Epsilon(pref, probsame, probsame[ids])
#print('epsilon: {}'.format(e.score()))

'''
e = Epsilon(pref, probsame, w1)
#e1 = Epsilon1(pref, probsame, w2)
print('epsilon: {}'.format(e.score()))
e = Epsilon(pref, probsame, w2)
print('epsilon: {}'.format(e.score()))
'''
#print('epsilon1: {}'.format(e1.score()))
'''
rel = Relevance(scores, b, func=lambda w, s: s)
print('rel_v1: {}'.format(rel.score(v1)))
print('rel_v2: {}'.format(rel.score(v2)))
'''
'''

'''


'''

start = time.time()
alg_greedy = Greedy(scores, probsame, b, rel_func=lambda w,s: s)
result_greedy = alg_greedy.run(pref=pref, alpha=0)
print('greedy solution: {}'.format(result_greedy))
print('runtime: {}'.format(time.time()-start))
#dice = metrics.rank.dice_similarity(result_exhaustive, result_greedy)
#print('dice similarity: {}'.format(dice))
'''

'''
start = time.time()
alg_exhaustive = Exhaustive(scores, probsame, b, rel_func=lambda w,s: s)

result_exhaustive, _ = alg_exhaustive.run(pref=pref, alpha=0)
print('exhaustive solution: {}'.format(result_exhaustive))
#result_exhaustive, _ = alg_exhaustive.run(pref=pref, alpha=0)
#print('exhaustive solution original: {}'.format(result_exhaustive))
#print(result_exhaustive == result_exhaustive1)
print('runtime: {}'.format(time.time()-start))


alg_greedy = Greedy(scores, probsame, b)
result_greedy = alg_greedy.run1(pref=pref, alpha=0)
print('greedy solution: {}'.format(result_greedy))
dice = metrics.rank.dice_similarity(result_exhaustive, result_greedy)
print('dice similarity: {}'.format(dice))
'''

'''
rec_ids = np.nonzero(scores >= b)
scores_rec = scores[rec_ids]
probsame_rec = probsame[rec_ids]
'''
'''
start = time.time()
e = Epsilon(pref, probsame, probsame)
multiplier = 10
probsame = (probsame * multiplier).astype(int)
target = int(3.94 * multiplier)
closest_sum = e.subset_sum(probsame, target) / multiplier
print('closest_sum: {}'.format(closest_sum))
print('runtime: {}'.format(time.time()-start))
'''


rec_ids = np.nonzero(scores >= b)
scores_rec = scores[rec_ids]
probsame_rec = probsame[rec_ids]

'''
start = time.time()
e = Epsilon(pref, probsame, probsame_rec)
print ('epsilon: {0:.2f}'.format(e.score()))
print('runtime e: {}'.format(time.time()-start))
start = time.time()
e1 = Epsilon1(pref, probsame, probsame_rec, prec_knapsack=2)
print ('epsilon1: {0:.2f}'.format(e1.score()))
print('runtime e1: {}'.format(time.time()-start))
'''
#lower_bound, upper_bound = e.alpha_bounds(alpha=0)
#print('rec_set_size: {}'.format(scores_rec.size))
#print('lower bound: {0}, upper bound: {1}'.format(lower_bound*scores_rec.size, upper_bound*scores_rec.size))

#print ('epsilon original: {0:.2f}'.format(e.score()))

#print('alpha subsets: {}'.format(e.alpha_subsets(0.0)))


print('m_cand: {}'.format(np.mean(probsame)))
print('m_rec: {}'.format(np.mean(probsame_rec)))
#print ('epsilon: {0:.2f}'.format(e.score()))
#print('m_star: {}'.format(e.m_star()))

'''
start = time.time()
delta_min, delta_max, _, _ = e.delta_bounds()
print('d_min (old): {}'.format(delta_min))
print('d_max (old):{}'.format(delta_max))
print('epsilon: {}'.format(e.score()))
print('runtime: {}'.format(time.time()-start))

start = time.time()
delta_min, delta_max = e1.delta_bounds()
print('d_min (new): {}'.format(delta_min))
print('d_max (bew):{}'.format(delta_max))
print('epsilon: {}'.format(e1.score()))
print('runtime: {}'.format(time.time()-start))
'''
#e.visualize(prefs=[0.0, 0.647, 1.0])
