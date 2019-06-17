# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:28:03 2019

@author: ae0670
"""

import numpy as np
from metrics.epsilon import Epsilon
from metrics.relevance import Relevance
import sys

class Knapsack:
    
    def __init__(self, scores, probsame, b=0.5, rel_func=lambda x,y: x*y, multiplier=1):
        assert(multiplier in {1,10,100,1000,10000})
        sort = sorted(zip(scores, probsame))
        self.scores, self.probsame = map(np.array, zip(*sort))
        self.b = b
        self.multiplier = multiplier
        self.split_index = np.argmax(self.scores >= b) # find the score to split on
        self.scores_rec = self.scores[self.split_index:]
        self.probsame_rec = self.probsame[self.split_index:] * self.multiplier
        self.probsame_rec = self.probsame_rec.astype(int)
        self.rel = Relevance(self.scores, self.b, func=rel_func)
        self.probsame = (self.probsame * self.multiplier).astype(int)
        
    
    def run1(self, pref, alpha):
        total_capacity = int(self.probsame.size * self.multiplier)
        matrix = [[0]*(total_capacity+1) for _ in range(self.probsame.size+1)]
        for i, weight in enumerate(self.probsame, 1):
            for j in range(total_capacity+1):
                if weight > j:
                    matrix[i][j] = matrix[i-1][j] #item not included
                else:
                    no_add = matrix[i-1][j]
                    do_add = matrix[i-1][j-weight]+self.scores[i-1]
                    matrix[i][j] = max(no_add, do_add)
        # Reconstruction
        best_value = -sys.float_info.max
        best_capacity = 0
        best_subset = None
        for i in range(1, self.probsame.size):
            e = Epsilon(pref, self.probsame, np.zeros(i))
            m_lower, m_upper = e.alpha_bounds(alpha)
            min_capacity, max_capacity = int(m_lower*i), int(m_upper*i)
            #value = matrix[self.probsame.size][max_capacity]
            #if value > best_value:
            subset, subset_weights, subset_capacity = self.reconstruct1(matrix, min_capacity, max_capacity, i)
            '''
            if subset is not None:
                print ('solution for size: {}'.format(i))
                print ('scores: {}'.format(subset))
                print ('weights: {}'.format(subset_weights/self.multiplier))
                print ('value: {}'.format(sum(subset-self.b)))
            '''
            #if subset_capacity >= min_capacity:
                    #best_value = value
                    #best_subset = subset
        #return best_subset, best_value
        return None, None
    
    def reconstruct1(self, matrix, min_capacity, max_capacity, target_size):
        subset = []
        subset_weights = []
        j = max_capacity
        for i in reversed(range(1, self.probsame.size+1)):
            if matrix[i][j] != matrix[i-1][j]:
                subset.append(self.scores[i-1])
                subset_weights.append(self.probsame[i-1])
                j -= self.probsame[i-1]
                #if len(subset) == target_size:
                #    break
        subset.reverse()
        subset_weights.reverse()
        subset_capacity = max_capacity-j
        '''
        for i in range(len(subset)):
            if subset_capacity-subset_weights[0] >= min_capacity:
        '''
        i = 0
        while len(subset) > target_size:
            if subset_capacity-subset_weights[i] >= min_capacity:
                del subset[i]
                del subset_weights[i]
                subset_capacity = sum(subset_weights)
            else:
                i += 1
        
        return np.array(subset), np.array(subset_weights), subset_capacity
        '''
        if subset_capacity >= min_capacity:
            return np.array(subset), np.array(subset_weights), subset_capacity
        else:
            return None, 0
        '''


    '''original'''
    
    
    def run(self, pref, alpha, total_capacity=None, capacity_interval=None):
        if not total_capacity:
            total_capacity = self.probsame.size #self.probsame_rec.size
            #total_capacity = np.sum(self.probsame_rec)
        total_capacity = int(total_capacity * self.multiplier)
        matrix = [[0]*(total_capacity+1) for _ in range(self.probsame.size+1)]
        for i, weight in enumerate(self.probsame, 1):
            for j in range(total_capacity+1):
                if weight > j:
                    matrix[i][j] = matrix[i-1][j] #item not included
                else:
                    no_add = matrix[i-1][j]
                    do_add = matrix[i-1][j-weight]+self.scores[i-1]
                    matrix[i][j] = max(no_add, do_add)
        # Reconstruction
        best_subset = []
        max_capacity = int(capacity_interval[1] * self.multiplier)
        min_capacity = int(capacity_interval[0] * self.multiplier)
        '''
        best_subset, subset_capacity, subset_value = self.reconstruct(matrix, max_capacity)
        print('best subset: {0}, capacity: {1}, value: {2}'.format
                  (best_subset, subset_capacity/self.multiplier, subset_value))
        '''
        for capacity in range(min_capacity, max_capacity+1):
            best_subset, subset_capacity, subset_value = self.reconstruct(matrix, capacity)
            if subset_capacity >= min_capacity:
                print('best subset: {0}, capacity: {1}, value: {2}'.format
                  (best_subset, subset_capacity/self.multiplier, subset_value))
        return np.array(best_subset), subset_value    
    
    def reconstruct(self, matrix, capacity):
        best_subset = []
        j = capacity
        for i in reversed(range(1, self.probsame.size+1)):
            if matrix[i][j] != matrix[i-1][j]:
                best_subset.append(self.scores[i-1])
                j -= self.probsame[i-1]
        best_subset.reverse()
        subset_value = matrix[self.probsame.size][capacity]
        subset_capacity = capacity-j
        #print('subset_capacity: {}'.format(subset_capacity/self.multiplier))
        return best_subset, subset_capacity, subset_value
    
    
    '''
    j = total_capacity
    for i in reversed(range(1, self.probsame_rec.size+1)):
        if matrix[i][j] != matrix[i-1][j]:
            best_subset.append(self.scores_rec[i-1])
            j -= self.probsame_rec[i-1]
    best_subset.reverse()
    return np.array(best_subset), matrix[self.probsame_rec.size][total_capacity]
    '''
    
    
    
    
    '''    
    for capacity in range(min_capacity, max_capacity+1):
        best_subset, subset_capacity, subset_value = self.reconstruct(matrix, capacity)
        if subset_capacity >= min_capacity:
            print('best subset: {0}, capacity: {1}, value: {2}'.format
              (best_subset, subset_capacity/self.multiplier, subset_value))
    return np.array(best_subset), subset_value
    '''