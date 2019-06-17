# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:29:35 2018

@author: ae0670
"""

import numpy as np
from metrics.epsilon import Epsilon
from metrics.epsilon_ss import Epsilon_ss
from metrics.relevance import Relevance
import sys
from itertools import combinations
from itertools import tee
from collections import Counter
import heapq


# For alpha=0, the best subset can be retrieved directly and very quickly from Epsilon.
# However, there can be several elements with same values of probsame,
# so then the goal is to pick the ones with the highest score.
# This, however, has to be done for all sizes in order to find the best solution.


class Exhaustive:
    
    def __init__(self, pref, scores, probsame, b=0.5, rel_func=lambda x,y: x*y):
        self.scores = scores
        self.probsame = probsame
        self.pref = pref
        self.b = round(b, 3)
        self.rel = Relevance(self.scores, self.b, func=rel_func)
        self.counters = list()
    
    # new: extracts only feasible subsets from epsilon
    def run(self, alpha, pref=None):
        if pref is None:
            pref = self.pref
        best_subset = [-sys.float_info.max]
        worst_subset = [sys.float_info.max]        
        for k in range(1, self.probsame.size+1):
                e = Epsilon(pref, self.probsame, np.zeros(k))
                subsets = list(map(self.scores_subset, e.alpha_subsets(alpha)))
                subsets = [s for s in subsets if s is not None]

                # ref_set = np.array([0.51, 0.86, 0.97])
                # #temp
                # if k == 3:
                #     for subset in subsets:
                #         if sorted(subset) == sorted(ref_set):
                #             print("**************************")
                #             print('found in exhaustive: {}'.format(subset))
                #             print("**************************")

                if subsets == []:
                    continue
                local_best = max(subsets, key=lambda s: self.rel.score(np.array(s)))
                local_worst = min(subsets, key=lambda s: self.rel.score(np.array(s)))
                best_subset = max([local_best, best_subset], key=lambda s: self.rel.score(np.array(s)))
                worst_subset = min([local_worst, worst_subset], key=lambda s: self.rel.score(np.array(s)))
        #best_subset = [x+self.b for x in best_subset]
        #worst_subset = [x+self.b for x in worst_subset]
        return np.array(sorted(best_subset)), np.array(sorted(worst_subset))
        
    def scores_subset(self, probsame_subset):
        counter = Counter(probsame_subset)
        if counter not in self.counters:
            self.counters.append(counter)
            subset = list()
            for c in counter.most_common():
                scores = self.scores[np.nonzero(self.probsame == c[0])]
                subset.extend(heapq.nlargest(c[1], scores))
            return subset
        return None
        
   # original: generates combinations
    def run1(self, pref, alpha):
        max_rel = -sys.float_info.max
        min_rel = sys.float_info.max
        best_subset = None
        worst_subset = None
        for k in range(1, self.probsame.size+1):
            #print('generating combinations...')
            subset_iter, subset_iter1 = tee(combinations(range(self.probsame.size), k))
            #print('done')
            #subsets = np.asarray(list(map(self.get_probsame_subset, subset_iter1))) 
            e = Epsilon(pref, self.probsame, np.zeros(k))
            for ids_tuple in subset_iter:
                probsame_subset = np.asarray(self.probsame[list(ids_tuple)])
                epsilon = e.score(rec=probsame_subset)
                #print ('subset {0}, epsilon = {1:.2f}, alpha-fair: {2}, score: {3}'
                #   .format(np.asarray(self.probsame[list(ids_tuple)]),epsilon,epsilon<=alpha,
                #           self.get_cost(self.scores[list(ids_tuple)])))
                if round(epsilon, 5) <= round(alpha, 5):
                #if epsilon <= alpha:
                    scores_subset = self.scores[list(ids_tuple)]
                    rel_score = self.rel.score(scores_subset)
                    if rel_score >= max_rel:
                        max_rel = rel_score
                        best_subset = scores_subset
                    if rel_score < min_rel:
                        min_rel = rel_score
                        worst_subset = scores_subset
        return np.array(sorted(best_subset)), np.array(sorted(worst_subset))

    def get_probsame_subset(self, ids_tuple):
        return self.probsame[list(ids_tuple)]