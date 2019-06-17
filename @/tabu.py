# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:33:51 2019

@author: ae0670
"""
import numpy as np
from metrics.epsilon import Epsilon
from metrics.relevance import Relevance
import sys

class Tabu:
    
    def __init__(self, pref, scores, probsame, b=0.5, tabu_size=2, rel_func=lambda x,y: x*y):
        sort = sorted(zip(scores, probsame))
        self.scores, self.probsame = map(np.array, zip(*sort))
        self.b = b
        self.pref = pref
        self.rel = Relevance(self.scores, self.b, func=rel_func)
        self.rec_ids = np.nonzero(self.scores >= b)[0]
        self.pool_ids = np.nonzero(self.scores < b)[0]
        self.scores_rec = self.scores[self.rec_ids]
        self.probsame_rec = self.probsame[self.rec_ids]
        self.current = self.rec_ids.tolist()
        self.pool = self.pool_ids.tolist()
        self.epsilons = [Epsilon(self.pref, self.probsame, np.zeros(i)) 
                            for i in range(probsame.size+1)]
        self.tabu_list = []
        self.solution_list = []
        self.tabu_size = tabu_size
        self.iterations = 0
        #self.tabu_add = [self.current[-1]]
        
        self.split_index = np.argmax(self.scores >= b) # find the score to split on
        self.scores_rec = self.scores[self.split_index:]
        self.probsame_rec = self.probsame[self.split_index:]
        
    def is_tabu(self, item):
        if item in self.tabu_list:
            return True
        sign_delta = np.sign(self.m_current()-self.epsilons[len(self.current)].m_star())
        sign_operation = np.sign(self.m_current()-self.probsame[item])
        if item in self.current: # evaluate deletion: signs should not match
            return sign_delta == sign_operation or len(self.current) == 1
        else: # evaluate addition: signs should match
            return sign_delta != sign_operation
        
    def is_admissible(self, item):
        sign_delta = np.sign(self.m_current()-self.epsilons[len(self.current)].m_star())
        sign_operation = np.sign(self.m_current()-self.probsame[item])
        if item in self.current: # evaluate deletion: signs should not match
            return sign_delta == sign_operation or len(self.current) == 1
        else: # evaluate addition: signs should match
            return sign_delta != sign_operation
            
    def m_current(self):
        return np.mean(self.probsame[self.current])
    
    def next_item(self, from_list):
        for _, item in enumerate(from_list):
            if not self.is_tabu(item):
                return item
    
    
    '''
    def next_item(self, from_list):
        candidate = []
        for _, item in enumerate(from_list):
            #if item not in self.tabu_list:
            solution = self.solution(item)
            if solution not in self.solution_list: # and len(solution) > 0:
                if len(candidate) == 0:
                    candidate.append(item)
                if self.is_admissible(item):
                    return item
        try:
            return candidate[0]
        except:
            return None
    '''
        
        
    '''
    def next_item(self, from_list):
        candidate = []
        for _, item in enumerate(from_list):
            if item not in self.tabu_list:
                if len(candidate) == 0:
                    candidate.append(item)
                if self.is_admissible(item):
                    return item
        try:
            return candidate[0]
        except:
            return None
    '''
    
    def get_neighbors(self):
        neighborhood = []
        deletion_candidate = self.next_item(self.current)
        addition_candidate = self.next_item(reversed(self.pool))
        if deletion_candidate is not None and len(self.current) > 1:
            neighborhood.append(deletion_candidate)
        if addition_candidate is not None:
            neighborhood.append(addition_candidate)
        return neighborhood
    
    def pick_best(self, neighborhood):
        best_item = None
        best_solution = None
        best_fitness = -sys.float_info.max
        for item in neighborhood:
            solution = self.solution(item)
            fitness = self.fitness(solution)
            if fitness >= best_fitness:
                best_fitness = fitness
                best_item = item
                best_solution = solution
        return best_item, best_solution
    
    def solution(self, item):
        solution = list(self.current)
        if item in self.current:
            solution.remove(item)
        else:
            solution.insert(0, item)
            if len(self.current) == 1: #and solution in self.solution_list: # consider substitution
                solution = [item]
        return solution

    def fitness(self, subset):
        #return 1-self.epsilon(subset) * self.rel.score(self.scores[subset])
        return self.rel.score(self.scores[subset])
    
    def epsilon(self, subset):
        try:
            return self.epsilons[len(subset)].score(rec=self.probsame[subset])
        except:
            print('\nscores: {}'.format(self.scores))
            print('probsame: {}'.format(self.probsame))
            print('pref: {}'.format(self.pref))
            print('pool: {}'.format(self.pool))
            print('current: {}'.format(self.current))
    
    def update_pool(self, item):
        if item in self.pool:
            self.pool.remove(item)
        else:
            self.pool.append(item)
        return
    
    def restart(self):
        self.iterations = 0
        self.current = self.rec_ids.tolist()
        self.pool = self.pool_ids.tolist()
        self.tabu_list = []
    
    def run(self, alpha, pref=None):
        if pref is None:
            pref = self.pref
        #best_global = self.current
        #best_candidate = best_global
        best_item = None
        while round(self.epsilon(self.current), 2) > round(alpha, 2):
        #while self.epsilon(self.current) > alpha:
            self.iterations += 1
            neighborhood = self.get_neighbors()
            '''
            if len(neighborhood) == 0:
                self.restart()
                print('restarting')
                continue
            '''
            best_item, best_candidate = self.pick_best(neighborhood)
            self.solution_list.append(list(self.current))
            self.current = best_candidate
            self.update_pool(best_item)
            self.tabu_list.append(best_item)
            if len(self.tabu_list) > self.tabu_size:
                del self.tabu_list[0]
            #print('iteration {}'.format(iterations))
        
            if self.iterations == 500:
                print('limit reached, restarting...')
                self.restart()
                '''
                self.print_vector(self.scores, 'scores')
                self.print_vector(self.probsame, 'probsame')
                print('pref: {}'.format(self.pref))
                '''
                continue
        
        #print('iterations: {}'.format(iterations))
        if len(self.current) == 0:
            self.print_vector(self.scores, 'scores')
            self.print_vector(self.probsame, 'probsame')
            print('pref: {}'.format(self.pref))
        return self.scores[self.current] 
    
        '''
        for item in neighborhood:
            candidate = self.solution(item)
            if self.fitness(candidate) > self.fitness(best_candidate):
                best_candidate = candidate
                best_item = item
        if self.fitness(best_candidate) > self.fitness(best_global):
            best_global = best_candidate
        if best_item not in neighborhood:
            best_item, best_candidate = self.pick_best(neighborhood)
        '''
    def print_vector(self, vector, title):
        print(f"{title}: ", sep='', end='', flush=True)
        for i in vector:
            print(f"{i}, ", sep=',', end='', flush=True)
        print()