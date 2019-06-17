# -*- coding: utf-8 -*-
import sys
sys.path.append('..')  # to include from parent's directory
import numpy as np
from metrics.epsilon import Epsilon


"""
Experiment 1.
Visualize the fairness of input data for various values of pref
"""

scores_cand = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
probsame_cand = np.array([0, 0, 1, 0, 1, 0, 0, 1, 1, 1])

output_path = 'plots/'
b = 0.5
prefs = [0.0, 0.5, 1.0]

rec_ids = np.nonzero(scores_cand >= b)
scores_rec = scores_cand[rec_ids]
probsame_rec = probsame_cand[rec_ids]


e = Epsilon(0, probsame_cand, probsame_rec)
print('epsilon: {0:.2f}'.format(e.score()))
e.visualize(prefs)