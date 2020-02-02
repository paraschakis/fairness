# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:33:51 2019

@author: ae0670
"""
import numpy as np
from metrics.epsilon import Epsilon
from metrics.epsilon_ss import Epsilon_ss
from metrics.relevance import Relevance
import sys


class Tabu:
    """
    Tabu search
    """

    def __init__(self, pref, scores, probsame, b=0.5, 
                 tabu_size=2, return_ids=False, rel_func=lambda x, y: x * y):
        self.scores = np.copy(scores)
        self.probsame = np.copy(probsame)
        
        self.return_ids = return_ids
        self.b = b
        self.pref = pref
        self.rel = Relevance(self.scores, self.b, func=rel_func)
        
        rec_ids = np.nonzero(self.scores >= b)[0]
        pool_ids = np.nonzero(self.scores < b)[0]
        
        sort = sorted(zip(scores, rec_ids))
        _, rec_ids = map(np.array, zip(*sort))
        self.current = rec_ids.tolist()
        try:
            sort = sorted(zip(scores, pool_ids))
            _, pool_ids = map(np.array, zip(*sort))
            self.pool = pool_ids.tolist()
        except ValueError: # fair but uninformative recommendation -> no pool
            self.pool = []
            
        self.epsilons = [Epsilon(self.pref, self.probsame, np.zeros(i))
                         for i in range(probsame.size + 1)]
        self.tabu_list = []
        self.rec_ids = rec_ids
        self.pool_ids = pool_ids
        self.tabu_size = tabu_size
        self.iterations = 0

    def is_tabu(self, item):
        if item in self.tabu_list:
            return True
        sign_delta = np.sign(self.m_current() - self.epsilons[len(self.current)].m_star())
        sign_operation = np.sign(self.m_current() - self.probsame[item])
        if item in self.current:  # evaluate deletion: signs should not match
            return sign_delta == sign_operation or len(self.current) == 1
        else:  # evaluate addition: signs should match
            return sign_delta != sign_operation

    def next_item(self, from_list):
        for _, item in enumerate(from_list):
            if not self.is_tabu(item):
                return item

    def get_neighbors(self):
        neighborhood = []
        deletion_candidate = self.next_item(self.current)
        addition_candidate = self.next_item(reversed(self.pool))
        if deletion_candidate is not None:  # and len(self.current) > 1:
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
        return solution

    def m_current(self):
        return np.mean(self.probsame[self.current])

    def fitness(self, subset):
        return self.rel.score(self.scores[subset])

    def epsilon(self, subset):
        return self.epsilons[len(subset)].score(rec=self.probsame[subset])

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
        self.tabu_size += 1
        print('trying with tabu_size = {}'.format(self.tabu_size))

    def run(self, alpha, pref=None):
        while round(self.epsilon(self.current), 2) > round(alpha, 2):
            self.iterations += 1
            neighborhood = self.get_neighbors()
            best_item, best_candidate = self.pick_best(neighborhood)
            self.current = best_candidate
            self.update_pool(best_item)
            self.tabu_list.append(best_item)
            if len(self.tabu_list) > self.tabu_size:
                del self.tabu_list[0]

            if self.iterations == 100:
                print('limit reached, restarting...')
                self.restart()
                continue

        if self.return_ids:
            return sorted(self.current)
        else:
            return self.scores[self.current]
