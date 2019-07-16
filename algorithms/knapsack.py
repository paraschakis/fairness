# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:58:58 2019

@author: ae0670
"""

import numpy as np
from metrics.epsilon import Epsilon
from metrics.epsilon_ss import Epsilon_ss
from metrics.relevance import Relevance
import sys
import heapq

np.random.seed = 7

class Knapsack:
    def __init__(self, pref, scores, probsame, 
                 b=0.5, prec=2, return_ids=False, rel_func=lambda x, y: x * y):
        self.return_ids = return_ids
        self.scores = np.copy(scores)
        self.probsame = np.copy(probsame)
        self.b = b
        self.pref = pref
        self.prec = prec
        multiplier = 10**self.prec

        self.rec_ids = np.nonzero(self.scores >= b)
        self.scores_rec = self.scores[self.rec_ids]
        self.probsame_rec = np.rint((probsame[self.rec_ids]*multiplier)).astype(int)
        self.probsame = np.rint((self.probsame*multiplier)).astype(int)
        self.rel = Relevance(self.scores, self.b, func=rel_func)
        
    def knapsack(self, n, k, l, u):
        """
        3-dimensional knapsack with cardinality and capacity constraints
        
        Inputs:
            n: number of items
            k: cardinality of solution
            l: lower bound on capacity
            u: upper bound on capacity
            
        Returns k-sized sequence that maximizes the score under constraints,
        or empty sequence if no solution exists
        """
        if n == 0 or k == 0:
            return 0
        if self.T[n,u,k] != 0:
            return self.T[n,u,k]
        v1 = 0
        v2 = 0
        if self.probsame[n-1] <= u and \
            sum(heapq.nlargest(k-1, self.probsame[:n-1])) >= l-self.probsame[n-1]:
            v1 = self.knapsack(n-1, k-1, l-self.probsame[n-1], 
                               u-self.probsame[n-1])+self.scores[n-1]
        if sum(heapq.nlargest(k, self.probsame[:n-1])) >= l:
            v2 = self.knapsack(n-1, k, l, u)
        self.T[n,u,k] = max(v1, v2)         
        return self.T[n,u,k]

    def run(self, alpha, pref=None):
        # check if the rec.set is already fair
#        e = Epsilon(pref, self.probsame, self.probsame_rec)
#        if e.score() <= alpha:
#            if self.return_ids:
#                return sorted(self.rec_ids)
#            else:
#                return self.scores[self.rec_ids] 
        
        if pref is None:
            pref = self.pref
        m = sum(self.scores) # additive constant to enforce k-sized solutions
        self.scores += m
        best_ids = None
        best_rel = -sys.float_info.max
        for k in range(1, self.probsame.size+1):
            n = self.probsame.size
            e = Epsilon(pref, self.probsame, self.probsame[:k])
#            e = Epsilon1(pref, self.probsame, self.probsame[:k], prec_numeric=13, prec_knapsack=1)
            m_lower, m_upper = e.alpha_bounds(alpha)
            l = round(m_lower*k).astype(int) # lowest capacity
            u = round(m_upper*k).astype(int) # highest capacity
            self.T = np.zeros((n+1,u+1,k+1), dtype=float)
            self.knapsack(n, k, l, u)
            # reconstruct
            solution_ids = []
            #print('solution for k = {}'.format(k))
            if self.T[n,u,k] >= k*m: #check if solution exists
                for n in reversed(range(1, self.probsame.size+1)):
                    if self.T[n,u,k] != self.T[n-1,u,k]:
                        solution_ids.append(n-1)
                        u -= self.probsame[n-1]
                        k -= 1
            #print(solution)
            if len(solution_ids) > 0:
                rel = self.rel.score(self.scores[solution_ids]-m)
                if rel > best_rel:
                    best_rel = rel
                    best_ids = solution_ids
        if self.return_ids:
            return sorted(best_ids)
        else:
            return self.scores[best_ids]-m
#        return np.asarray(sorted(best_solution))


            
        '''
        e = Epsilon(pref, self.probsame, self.probsame_rec)
        m_lower, m_upper = e.alpha_bounds(alpha) # lower and upper bound
        k = self.probsame_rec.size #cardinality constraint
        n = self.probsame.size
        m = sum(self.scores) # number to add to scores (for exact-k KP)
        self.scores += m
        l = round(m_lower*k).astype(int) # lowest capacity
        u = round(m_upper*k).astype(int) # highest capacity
        self.T = [[[0 for _ in range(k+1)] for _ in range(u+1)] for _ in range(n+1)]
        self.knapsack(n, k, l, u)
        # reconstruct
        solution = []
        if self.T[n][u][k] >= k*m: #check if solution exists
            for n in reversed(range(1, self.probsame.size+1)):
                if self.T[n][u][k] != self.T[n-1][u][k]:
                    solution.append(self.scores[n-1]-m)
                    u -= self.probsame[n-1]
                    k -= 1
        #print(self.T)
        print(self.rel.score(np.asarray(solution)))
        return np.asarray(solution)
        '''