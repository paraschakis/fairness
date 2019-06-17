# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:25:04 2018

@author: ae0670

"""

import numpy as np
from epsilon import Epsilon
import sys
import itertools

class Scorer:
    
    # Inputs:
    # scores: output probabilities of a recommender system
    # probsame: probabilities of the same race/religion
    # b: decision boundary
    
    def __init__(self, scores, probsame, b=0.5):
        self.scores = scores
        self.probsame = probsame
        self.b = b
        self.split_index = np.argmax(scores >= b) # find the score to split on
        self.scores_rec = scores[self.split_index:]
        self.probsame_rec = probsame[self.split_index:]

    # Relevance score to be maximized
    def score(self, subset=None):
        if subset is None:
            subset = self.scores_rec
        w = min(self.scores_rec.size, subset.size)/max(self.scores_rec.size, subset.size)
        s = sum(subset-self.b)
        return w*s

    # Find optimal subset that satisfies fairness constraints and maximizes relevance
    def exhaustive_search(self, pref, alpha):
        max_rel = sys.float_info.min
        best_subset = None
        for k in range(1, self.probsame.size+1): 
            subset_ids = itertools.combinations(range(self.probsame.size), k)
            e = Epsilon(pref, self.probsame, np.zeros(k))
            for ids_tuple in subset_ids:
                probsame_subset = np.asarray(self.probsame[list(ids_tuple)])
                epsilon = e.score(rec=probsame_subset)
                #print ('subset {0}, epsilon = {1:.2f}, alpha-fair: {2}, score: {3}'
                #   .format(np.asarray(self.probsame[list(ids_tuple)]),epsilon,epsilon<=alpha,
                #           self.get_cost(self.scores[list(ids_tuple)])))
                if epsilon <= alpha:
                    scores_subset = self.scores[list(ids_tuple)]
                    rel = self.score(scores_subset)
                    if rel >= max_rel:
                        max_rel = rel
                        best_subset = scores_subset
        return best_subset
    
    def greedy_search(self, pref, alpha, scores_current=None, probsame_current=None,
                      scores_rest=None, probsame_rest=None):
        if scores_current is None:
            scores_current = self.scores_rec
            scores_rest = self.scores[:self.split_index]
        if probsame_current is None:
            probsame_current = self.probsame_rec
            probsame_rest = self.probsame[:self.split_index]
            if self._alpha_fair(pref, alpha, probsame_current):
                return scores_current                
        e = Epsilon(pref, self.probsame, probsame_current)
        m_current = np.mean(probsame_current)
        # addition
        rel_after_add = -sys.float_info.min
        for i in range(probsame_rest.size):
            if np.sign(m_current-probsame_rest[::-1][i]) == np.sign(m_current-e.m_star()):
                rel_after_add = self.score(np.append(scores_current, scores_rest[::-1][i]))
                break
        # deletion
        rel_after_del = -sys.float_info.min
        for j in range(probsame_current.size):
            if np.sign(m_current-probsame_current[j]) != np.sign(m_current-e.m_star()):
                rel_after_del = self.score(np.delete(scores_current, j))
                break
        if rel_after_add >= rel_after_del:
            scores_current = np.append(scores_current, scores_rest[::-1][i])
            probsame_current = np.append(probsame_current, probsame_rest[::-1][i])
            scores_rest = np.delete(scores_rest, scores_rest.size-i-1)
            probsame_rest = np.delete(probsame_rest, probsame_rest.size-i-1)
        else:
            scores_current = np.delete(scores_current, j)
            probsame_current = np.delete(probsame_current, j)           
        if not self._alpha_fair(pref, alpha, probsame_current):
            return self.greedy_search(pref, alpha, scores_current, probsame_current,
                                      scores_rest, probsame_rest)
        else:
            return scores_current
    
    def _alpha_fair(self, pref, alpha, subset):
        e = Epsilon(pref, self.probsame, subset)
        return e.score() <= alpha
    
    
'''
    # find the closest index that improves the mean
    def _next_index(self, array, m_current, m_star):
        for i in range(array.size):
            if np.sign(m_current-array[i]) == np.sign(m_current-m_star()):
                return i
        return -1
'''