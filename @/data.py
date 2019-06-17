# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:02:16 2018

@author: ae0670
"""

import itertools
import numpy as np

class Data():
    
    def __init__(self, scores, probsame):
        self.scores = np.array(scores)
        self.probsame = np.array(probsame)
        self.index_map = dict()
        self.probsame_map = dict()
        self._combinations()
        
    def _combinations(self):
        for k in range (1, self.scores.size+1):
            subsets = itertools.combinations(range(self.scores.size), k)
            self.index_map[k] = list(map(lambda x: np.array(x), subsets))
            self.probsame_map[k] = map(lambda x: np.array(self.probsame[x]), self.index_map[k])