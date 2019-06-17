# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:40:38 2018

@author: ae0670
"""
import numpy as np
import itertools

class Epsilon():
    
    def __init__(self):
        pass

    def epsilon(self, pref, cand_set, rec_set):
        m_max = np.mean(-np.sort(-cand_set)[:rec_set.size])
        m_cand = np.mean(cand_set)
        m_rec = np.mean(rec_set)
        m_star = m_cand+pref*(m_max-m_cand)
        delta_min, delta_max = self.get_delta_bounds(m_star, cand_set, rec_set.size)
        delta = abs(m_star-m_rec)
        return (delta-delta_min)/(delta_max-delta_min+1e-17)
    
    def get_delta_bounds(self, m_star, cand_set, subset_size):
        means = self.get_means(cand_set, subset_size)
        deltas = np.abs(means-m_star)
        return np.min(deltas), np.max(deltas)
    
    def get_means(self, cand_set, subset_size):
        subsets = itertools.combinations(cand_set, subset_size)
        subsets = np.asarray(list(dict.fromkeys(subsets))) # remove duplicates
        return np.asarray(list(map(np.mean, subsets)))
    
    #def get_max(self, subset):
    #    return np.mean(-np.sort(-self.cand_set)[:subset.size])
    
    