# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:55:52 2018

@author: ae0670
"""
import matplotlib.pyplot as plt
from epsilon import Epsilon as e


def visualize(prefs, cand_set, rec_set):
    plt.title('fairness penalty curve')
    plt.xlabel('mean(rec_set)')
    plt.ylabel('penalty')
    
    for pref in prefs:
        x, y = generate_plot_data(pref, cand_set, rec_set.size)
        
def generate_plot_data(pref, cand_set, size):
    x = []
    y = []
    f = e.get_means(cand_set, size)
    
    for i in np.argsort(subset_means):
        x.append(subset_means[i])
        y.append(get_penalty7(pref, subsets[i]))
    return x, y