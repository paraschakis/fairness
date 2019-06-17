# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:38 2018

@author: ae0670

Calculates epsilon measure, taking values from 0 (best) to 1 (worst).

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import heapq
from collections import defaultdict

class Epsilon_ss:
    
    """
    Epsilon class that uses 'subset sum' (Knapsack) approach for calculating the lower bound.
    Runs slower than exhaustive.
    """
     
    def __init__(self, pref, cand, rec, prec_numeric=13, prec_knapsack=1):
        self.pref = pref
        self.cand = cand
        self.rec = rec
        self.m_cand = np.mean(cand)
        sorted_cand = np.sort(cand)
        #self.m_min = np.mean(sorted_cand[:rec.size])
        #self.m_max = np.mean(sorted_cand[-rec.size:])
        self.m_min = round(np.mean(sorted_cand[:rec.size]), prec_numeric)
        self.m_max = round(np.mean(sorted_cand[-rec.size:]), prec_numeric)
        self.delta_min = None
        self.delta_max = None
        self.info = dict()
        self.prec_numeric = prec_numeric
        self.prec_knapsack = prec_knapsack

    def m_star(self, pref=None):
        if pref == None:
            pref = self.pref
        #return self.m_cand+pref*(self.m_max-self.m_cand)
        return round(self.m_cand+pref*(self.m_max-self.m_cand), self.prec_numeric)
     
    def fill_knapsack(self, n, c, k, v, w):
        for x in range(1, n+1):
            for y in range(0, c+1):
                for z in range(1, k+1):
                    if w[x-1] > y:
                        self.T[x,y,z] = self.T[x-1,y,z]
                    else:
                        self.T[x,y,z] = max(self.T[x-1,y,z], 
                              self.T[x-1,y-w[x-1],z-1] + v[x-1])
        return self.T[n,c,k]
    
    def reconstruct(self, n, c, k, w):
        solution = []
        #indices = []
        for i in reversed(range(1, n+1)):
            if self.T[i,c,k] != self.T[i-1,c,k]:
                #indices.append(i-1)
                solution.append(self.cand[i-1])
                #solution.append(np.mean(self.w_map[w[i-1]]))
                c -= w[i-1]
                k -= 1
        #return indices
        return solution    
    
    def closest_mean(self, m_star):
        """
        Finds the mean that is closest to m_star.
        Based on 'subset sum' algorithm (a variant of knapsack with weights=values)
        """
        multiplier = 10**self.prec_knapsack
        target_sum = m_star*self.rec.size*multiplier
        w = np.rint(self.cand*multiplier).astype(int)
        '''
        self.w_map = defaultdict(list)
        mapper = lambda x: self.w_map[np.rint(x*(10**self.prec)).astype(int)].append(x)
        list(map(mapper, self.cand))
        '''
        m = max(1, max(w)) # needed for size=1
        v = w+m
        n = self.cand.size
        k = self.rec.size
        c = sum(w)
        self.T = np.zeros((n+1,c+1,k+1), dtype=int)
        self.fill_knapsack(n, c, k, v, w)
        best_sum = -sys.float_info.max
        best_index = None
        neighbors = set()
        for j in range(c+1): # find capacity j that is closest to target
            if self.T[n,j,k] >= k*m:
                
                try: # needed only for prec=1
                    if self.prec_knapsack == 1 and self.T[n,j,k] != self.T[n,j-1,k]:
                        neighbors.add(j)
                except:
                    pass
                
                current_sum = self.T[n,j,k]-k*m
                if abs(target_sum-current_sum) < abs(target_sum-best_sum):
                    best_index = j
                    best_sum = current_sum
        #indices = self.reconstruct(n, best_index, k, w)
        #return np.mean(self.w[indices])
        #return np.mean(self.reconstruct(n, best_index, k, w))

        # for precision=1, choose from neighborhood, rather than single capacity
        if self.prec_knapsack == 1:
            neighborhood = heapq.nsmallest(3, 
                                           neighbors,
                                           key=lambda x: abs(x-best_index))
            best_mean = sys.float_info.max
            best_index = None
            for i in neighborhood:
                mean = np.mean(self.reconstruct(n, i, k, w))
                if abs(mean-m_star) < abs(best_mean-m_star):
                    best_mean = mean
                    best_index = i
            return best_mean
        else:
            return np.mean(self.reconstruct(n, best_index, k, w))
       
    def delta_bounds(self, m_star=None):
        if m_star == None:
            m_star = self.m_star()
        m_closest = self.closest_mean(m_star)
        self.delta_min = abs(m_star-m_closest)
        self.delta_max = max(abs(m_star-self.m_max), abs(m_star-self.m_min))
#        print('delta_min: {}'.format(self.delta_min))
#        print('delta_max: {}'.format(self.delta_max))
        '''
        if np.isnan(m_closest):
            print(f'm_closest is nan')
        '''    
        #return self.delta_min, self.delta_max
        return round(self.delta_min, self.prec_numeric), round(self.delta_max, self.prec_numeric)
        
    # returns lower and upper bounds on subset means defined by alpha
    def alpha_bounds(self, alpha):
        m_star = self.m_star()
        delta_min, delta_max = self.delta_bounds(m_star)
        delta_alpha = alpha*(delta_max-delta_min)+delta_min
        m_lower = max(m_star-delta_alpha, self.m_min)
        m_upper = min(m_star+delta_alpha, self.m_max)     
        '''
        if np.isnan(delta_min):
            delta_min = delta_max

            print(f'm_upper: {m_upper}')
            print(f'm_lower: {m_lower}')
            print(f'delta_min: {delta_min}')
            print(f'delta_max: {delta_max}')
            print(f'm_star: {m_star}')
            print(f'm_min: {self.m_min}')
            print(f'cand: {self.cand}')
        '''
        #return m_lower, m_upper
        return round(m_lower, self.prec_numeric), round(m_upper, self.prec_numeric)    
    
    def score(self, pref=None, rec=None, as_fairness=False):
        if rec is None:
            rec = self.rec
        assert rec.size == self.rec.size, \
            "this object only allows rec. set of size {}".format(self.rec.size)
        #m_rec = np.mean(rec)
        m_rec = round(np.mean(rec), self.prec_numeric)
        m_star = self.m_star(pref)
        delta_min, delta_max = self.delta_bounds(m_star)
        delta = abs(m_star-m_rec)
        if delta_min == delta_max or delta < delta_min:
            epsilon = 0
        else:
            epsilon = (delta-delta_min)/(delta_max-delta_min)
            #epsilon = round(epsilon, 4)
        #epsilon = (delta-self.delta_min)/(self.delta_max-self.delta_min+1e-17)
        self.info['m_star'] = m_star
        self.info['delta'] = delta
        self.info['epsilon'] = epsilon
        return (1-epsilon)*100 if as_fairness else epsilon
    