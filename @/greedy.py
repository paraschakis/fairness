# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:29:35 2018

@author: ae0670
"""

import numpy as np
from metrics.epsilon import Epsilon
from metrics.relevance import Relevance
import sys

class Greedy:
    
    def __init__(self, scores, probsame, b=0.5, rel_func=lambda x,y: x*y):
        sort = sorted(zip(scores, probsame))
        self.scores, self.probsame = map(np.array, zip(*sort))
        self.b = b
        self.split_index = np.argmax(self.scores >= b) # find the score to split on
        self.scores_rec = self.scores[self.split_index:]
        self.probsame_rec = self.probsame[self.split_index:]
        self.rel = Relevance(self.scores, self.b, func=rel_func)
               
    def run(self, pref, alpha, scores_current=None, probsame_current=None,
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
        best_rel = -sys.float_info.max
        best_operation = ''
        addition_doable = False
        deletion_doable = False
        
        '''
        print('----- new iteration -----')
        print('probsame_current: {}'.format(probsame_current))
        print('m_current: {}'.format(m_current))
        print('m_star: {}'.format(e.m_star()))
        '''
        # evaluate addition
        if probsame_rest.size > 0:
            for i in range(probsame_rest.size):
                #print('probsame_rest[::-1][i]: {}'.format(probsame_rest[::-1][i]))
                if np.sign(m_current-probsame_rest[::-1][i]) == np.sign(m_current-e.m_star()):
                    addition_doable = True
                    scores_add = np.append(scores_current, scores_rest[::-1][i])
                    best_rel = self.rel.score(scores_add)
                    best_operation = 'add'
                    break

        # evaluate deletion
        for j in range(probsame_current.size):
            if scores_current[j] >= self.b and \
                np.sign(m_current-probsame_current[j]) != np.sign(m_current-e.m_star()):
                deletion_doable = True
                if probsame_current.size > 1:
                    scores_del = np.delete(scores_current, j)
                    if self.rel.score(scores_del) > best_rel:
                        best_rel = self.rel.score(scores_del)
                        best_operation = 'del'  
                break
            
        # evaluate substitution
        if addition_doable and deletion_doable:
            scores_sub = np.delete(scores_current, j)
            scores_sub = np.append(scores_sub, scores_rest[::-1][i])
            if self.rel.score(scores_sub) > best_rel:
                best_rel = self.rel.score(scores_sub)
                best_operation = 'sub'
        
        # perform operation
        if best_operation == 'add':
            scores_current = scores_add
            probsame_current = np.append(probsame_current, probsame_rest[::-1][i])
            scores_rest = np.delete(scores_rest, scores_rest.size-i-1)
            probsame_rest = np.delete(probsame_rest, probsame_rest.size-i-1)
        elif best_operation == 'del':
            scores_current = scores_del
            probsame_current = np.delete(probsame_current, j)
        elif best_operation == 'sub':
            scores_current = scores_sub
            probsame_current = np.delete(probsame_current, j)
            probsame_current = np.append(probsame_current, probsame_rest[::-1][i])
            scores_rest = np.delete(scores_rest, scores_rest.size-i-1)
            probsame_rest = np.delete(probsame_rest, probsame_rest.size-i-1)
        else:
            print('no operation is feasible!')
            print('scores: {}'.format(self.scores))
            print('probsame: {}'.format(self.probsame))
            print('pref: {}'.format(pref))
            print('m_star: {}'.format(e.m_star()))
            print('m_current:{}'.format(m_current))
            raise SystemExit(0)
        if not self._alpha_fair(pref, alpha, probsame_current):
            return self.run(pref, alpha, scores_current, probsame_current,
                                          scores_rest, probsame_rest)
        return np.array(sorted(scores_current))
    
    
    def _alpha_fair(self, pref, alpha, subset):
        e = Epsilon(pref, self.probsame, subset)
        return e.score() <= alpha


    
    def run1(self, pref, alpha, scores_current=None, probsame_current=None,
                      scores_rest=None, probsame_rest=None):
        if scores_current is None:
            scores_current = self.scores_rec
            scores_rest = self.scores[:self.split_index]
        if probsame_current is None:
            probsame_current = self.probsame_rec
            probsame_rest = self.probsame[:self.split_index]
            if self._alpha_fair(pref, alpha, probsame_current):
                return scores_current           
        #probsame = np.concatenate((probsame_current, probsame_rest))
        e = Epsilon(pref, self.probsame, probsame_current)
        m_current = np.mean(probsame_current)
        m_star = e.m_star()
        best_rel = -sys.float_info.max
        best_operation = ''
        best_delta = sys.float_info.max
        addition_doable = False
        deletion_doable = False
        
        # evaluate addition
        best_local = sys.float_info.max
        if probsame_rest.size > 0:
            best_i = 0
            for i in range(probsame_rest.size):
                if np.sign(m_current-probsame_rest[::-1][i]) == np.sign(m_current-e.m_star()):
                    addition_doable = True
                    scores_add = np.append(scores_current, scores_rest[::-1][i])
                    probsame_add = np.append(probsame_current, probsame_rest[::-1][i])
                    best_rel = self.rel.score(scores_add)
                    e = Epsilon(pref, self.probsame, probsame_add)
                    delta = abs(e.m_star()-np.mean(probsame_add))
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                    #best_delta = abs(e.m_star()-np.mean(probsame_add))
                    best_operation = 'add'
                    #break
            
        # evaluate deletion
        best_local = sys.float_info.max
        for j in range(probsame_current.size):
            if probsame_current.size > 1:
                #if scores_current[j] >= self.b and \
                if np.sign(m_current-probsame_current[j]) != np.sign(m_current-e.m_star()):
                    scores_del = np.delete(scores_current, j)
                    probsame_del = np.delete(probsame_current, j)
                    #if self.rel.score(scores_del) > best_rel:
                    e = Epsilon(pref, self.probsame, probsame_del)
                    delta = abs(e.m_star()-np.mean(probsame_del))
                    if delta < best_local:
                        best_local = delta
                        best_j = j
                    '''
                    if delta < best_delta:
                        best_rel = self.rel.score(scores_del)
                        best_delta = delta
                        best_operation = 'del'
                        #break 
                    '''
            elif addition_doable: #substitution
                scores_sub = np.delete(scores_current, j)
                scores_sub = np.append(scores_sub, scores_rest[::-1][i])
                probsame_sub = np.delete(probsame_current, j)
                probsame_sub = np.append(probsame_sub, probsame_rest[::-1][i])
                e = Epsilon(pref, self.probsame, probsame_sub)
                delta = abs(e.m_star()-np.mean(probsame_sub))
                #if self.rel.score(scores_sub) > best_rel:
                if delta < best_delta:
                    best_rel = self.rel.score(scores_sub)
                    best_delta = delta
                    best_operation = 'sub'
        if best_local < best_delta:
            best_rel = self.rel.score(scores_del)
            best_delta = delta
            best_operation = 'del'
        
        # perform operation
        if best_operation == 'add':
            scores_current = scores_add
            probsame_current = np.append(probsame_current, probsame_rest[::-1][best_i])
            scores_rest = np.delete(scores_rest, scores_rest.size-best_i-1)
            probsame_rest = np.delete(probsame_rest, probsame_rest.size-best_i-1)
        elif best_operation == 'del':
            scores_current = scores_del
            probsame_current = np.delete(probsame_current, best_j)
        elif best_operation == 'sub':
            scores_current = scores_sub
            probsame_current = np.delete(probsame_current, j)
            probsame_current = np.append(probsame_current, probsame_rest[::-1][best_i])
            scores_rest = np.delete(scores_rest, scores_rest.size-best_i-1)
            probsame_rest = np.delete(probsame_rest, probsame_rest.size-best_i-1)
        else:
            print('no operation is feasible!')
            print('scores: {}'.format(self.scores))
            print('probsame: {}'.format(self.probsame))
            print('pref: {}'.format(pref))
            print('m_star: {}'.format(e.m_star()))
            print('m_current:{}'.format(m_current))
            raise SystemExit(0)
        if not self._alpha_fair(pref, alpha, probsame_current):
            return self.run1(pref, alpha, scores_current, probsame_current,
                                          scores_rest, probsame_rest)
        return scores_current    
        
    
    
    
    
        
    '''
    def run(self, pref, alpha, scores_current=None, probsame_current=None,
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
        best_rel = -sys.float_info.max
        best_scores_rest = scores_rest
        best_probsame_rest = probsame_rest
        addition_doable = False
        deletion_doable = False
        
        # addition
        #rel_after_add = -sys.float_info.max
        if probsame_rest.size > 0:
            for i in range(probsame_rest.size):
                if np.sign(m_current-probsame_rest[::-1][i]) == np.sign(m_current-e.m_star()):
                    addition_doable = True
                    best_scores = np.append(scores_current, scores_rest[::-1][i])
                    best_probsame = np.append(probsame_current, probsame_rest[::-1][i])
                    best_scores_rest = np.delete(scores_rest, scores_rest.size-i-1)
                    best_probsame_rest = np.delete(probsame_rest, probsame_rest.size-i-1)
                    best_rel = self.rel.score(best_scores)
                    break

        # deletion
        #rel_after_del = -sys.float_info.max
        for j in range(probsame_current.size):
            if np.sign(m_current-probsame_current[j]) != np.sign(m_current-e.m_star()):
                deletion_doable = True
                if probsame_current.size > 1:
                    result_set = np.delete(scores_current, j)
                    if self.rel.score(result_set) > best_rel:
                        best_rel = self.rel.score(result_set)
                        best_scores = result_set
                        best_probsame = np.delete(probsame_current, j)
                    break
            
        # substitution
        if addition_doable and deletion_doable:
            result_set = np.delete(scores_current, j)
            result_set = np.append(result_set, scores_rest[::-1][i])
            if self.rel.score(result_set) > best_rel:
                best_scores = result_set
                best_probsame = np.delete(probsame_current, j)
                best_probsame = np.append(best_probsame, probsame_rest[::-1][i])
                best_scores_rest = np.delete(scores_rest, scores_rest.size-i-1)
                best_probsame_rest = np.delete(probsame_rest, probsame_rest.size-i-1)

        try:
            if not self._alpha_fair(pref, alpha, best_probsame):
                return self.run(pref, alpha, best_scores, best_probsame,
                                          best_scores_rest, best_probsame_rest)
        except:
            print(f' error! scores_rest={scores_rest}, \
                      scores_current={scores_current}, \
                      scores={self.scores}, probsame={self.probsame}, \
                      pref={pref}')
        else:
            return best_scores
    '''
    

    
    
    '''        
    if rel_after_add >= rel_after_del:
        try:
            scores_current = np.append(scores_current, scores_rest[::-1][i])
            probsame_current = np.append(probsame_current, probsame_rest[::-1][i])
            scores_rest = np.delete(scores_rest, scores_rest.size-i-1)
            probsame_rest = np.delete(probsame_rest, probsame_rest.size-i-1)
        except:
            print(f' error! scores_rest={scores_rest}, \
                  scores_current={scores_current}, scores={self.scores}')
    else:
        scores_current = np.delete(scores_current[::-1], j)
        probsame_current = np.delete(probsame_current[::-1], j)  
    '''
    '''
    scores_current = best_scores
    probsame_current = best_probsame
    scores_rest = best_scores_rest
    probsame_rest = best_probsame_rest
    '''