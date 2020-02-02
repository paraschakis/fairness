# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:25:04 2018

@author: ae0670

"""

import numpy as np


class Relevance:

    def __init__(self, scores, b=0.5, func=lambda x, y: x * y):
        self.scores = scores  # prediction scores
        self.b = b  # classification threshold
        self.rec_size = np.count_nonzero(scores >= b)
        self.func = func

    # Relevance score to be maximized
    def score(self, subset):
        w = min(self.rec_size, subset.size) / max(self.rec_size, subset.size)
        s = sum(subset - self.b)
        return self.func(w, s)
