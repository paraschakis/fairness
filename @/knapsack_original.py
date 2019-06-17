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

class Knapsack:
    def __init__(self, pref, scores, probsame, b=0.5, prec=2, rel_func=lambda x, y: x*y):
        sort = sorted(zip(scores, probsame))
        self.scores, self.probsame = map(np.array, zip(*sort))
        self.b = b
        self.pref = pref
        self.prec = prec
        self.split_index = np.argmax(self.scores >= b) # find the score to split on
        self.scores_rec = self.scores[self.split_index:]
        multiplier = 10**self.prec
        self.probsame_rec = np.rint((self.probsame[self.split_index:]*multiplier)).astype(int)
        self.probsame = np.rint((self.probsame*multiplier)).astype(int)
        #self.scores = (self.scores*self.multiplier).astype(int)
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
        if pref is None:
            pref = self.pref
        m = sum(self.scores) # additive constant to enforce k-sized solutions
        self.scores += m
        best_solution = None
        best_score = -sys.float_info.max
        for k in range(1, self.probsame.size+1):
            n = self.probsame.size
            e = Epsilon(pref, self.probsame, self.probsame[:k])
            #e = Epsilon1(pref, self.probsame, self.probsame[:k], prec_numeric=13, prec_knapsack=1)
            m_lower, m_upper = e.alpha_bounds(alpha)
            l = round(m_lower*k).astype(int) # lowest capacity
            u = round(m_upper*k).astype(int) # highest capacity
            self.T = np.zeros((n+1,u+1,k+1), dtype=float)
            self.knapsack(n, k, l, u)
            # reconstruct
            solution = []
            #print('solution for k = {}'.format(k))
            if self.T[n,u,k] >= k*m: #check if solution exists
                for n in reversed(range(1, self.probsame.size+1)):
                    if self.T[n,u,k] != self.T[n-1,u,k]:
                        solution.append(self.scores[n-1]-m)
                        u -= self.probsame[n-1]
                        k -= 1
            #print(solution)
            if len(solution) > 0:
                score = self.rel.score(np.asarray(solution))
                if score >= best_score:
                    best_score = score
                    best_solution = solution
        return np.asarray(sorted(best_solution))


            
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