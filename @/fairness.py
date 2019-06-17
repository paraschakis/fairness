# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:38 2018

@author: ae0670
"""
import numpy as np

class Fairness():
    
    def __init__(self, cand_set, rec_set):
        self.cand_set = cand_set
        self.rec_set = rec_set
        self.m_cand = np.mean(cand_set)
        self.m_rec = np.mean(rec_set)
        self.m_max = self.get_max(rec_set)

    def epsilon(self, pref, subset=None):
        if subset is None:
            subset = self.rec_set
            m = self.m_rec
            m_max = self.m_max
        else:
            m = np.mean(subset)
            m_max = self.get_max(subset)
        m_star = self.m_cand+pref*(m_max-self.m_cand)
        #print('m_star={}'.format(m_star))
        delta_min, delta_max = self.get_delta_bounds(m_star, subset.size)
        delta = abs(m_star-m)
        #print ('delta-delta_min = {}'.format(delta-delta_min))
        #print ('delta_max-delta_min = {}'.format(delta_max-delta_min+1e-17))
        #penalty = (delta-delta_min)/(delta_max-delta_min+1e-17)
        #print ('penalty = {0:.2f}'.format(penalty))
        return (delta-delta_min)/(delta_max-delta_min+1e-17)
    
    def get_max(self, subset):
        return np.mean(-np.sort(-self.cand_set)[:subset.size])
    
    def get_delta_bounds(m_star, subset_size):
        if subset_size != imp_rec.size:
            _, means = combinations(subset_size)
            deltas = np.abs(means-m_star)
        else:
            deltas = np.abs(subset_means-m_star)
        return np.min(deltas), np.max(deltas)