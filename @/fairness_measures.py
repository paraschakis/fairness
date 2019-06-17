# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:47:59 2018

@author: ae0670
"""
import math
import numpy as np
from scipy.special import comb
import itertools
import matplotlib.pyplot as plt
import unique_combinations as unicomb
import sys
import time

# Input

pref_list = [0.0, 0.5, 1.0]
b = 0.5 #decision boundary
alpha = 0 #threshold for unfrairness (epsilon)S

#imp_cand = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
imp_cand = np.array([0,0,1,0,0,1,1,0,1,1])
#imp_rec = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
#imp_rec = np.array([0,0,1,1,1])
scores_cand = np.array([0.9,0.8,0.7,0.6,0.51,0.49,0.40,0.30,0.20,0.10])
rec_ids = np.where(scores_cand >= b)
scores_rec = scores_cand[rec_ids]
imp_rec = imp_cand[rec_ids]
#scores_rec = np.array([0.9,0.8,0.7,0.6,0.51])
#imp_cand = np.array([0.1,0.2,0.3,0.3,0.3,0.5,0.7,0.7,0.8,0.8])
#imp_rec = np.array([0.1,0.3,0.7,0.8,0.8])

# Common statistics
m_rec = np.mean(imp_rec)
m_cand = np.mean(imp_cand)
min_m = np.mean(np.sort(imp_cand)[:imp_rec.size])
max_m = np.mean(-np.sort(-imp_cand)[:imp_rec.size])
tp = scores_cand[scores_cand >= b]

subsets = itertools.combinations(imp_cand, imp_rec.size)
subsets = np.asarray(list(dict.fromkeys(subsets))) # remove duplicates
#subsets_iterator = unicomb.combinations(imp_cand, imp_rec.size)
#subsets = np.stack(subsets_iterator)
subset_means = np.asarray(list(map(np.mean, subsets)))

#imp_subsets_ids = itertools.combinations(range(scores_cand.size), scores_rec.size)
#imp_subsets_means = np.asarray(list(map(np.mean, ?)))

#for ind_tuple in subsets:
#    ind_list = list(ind_tuple)
#    print (scores_cand[ind_list])

# Init combinations


def get_diff(minuend, subtrahend):
    for element in subtrahend:
        i = np.where(minuend == element)[0][0] # get index of first match
        minuend = np.delete(minuend, i)
    return minuend

def supp(s, c): # calculates 'combinatorial support' supp(s|c)
    values_s, counts_s = np.unique(s, return_counts=True)
    result = 1
    for i in range(len(values_s)):
        count_c = (c == values_s[i]).sum()
        result *= comb(count_c, counts_s[i])
    return result

def max_supp(c, size): # calculates maximum support for a given set and a target size
    best_s = [] # set that maximizes supp(s|c)
    values, counts = np.unique(c, return_counts=True)
    for i in range(len(values)):
        r = counts[i] / len(c)
        n = min(int(r*size+0.5), size-len(best_s)) # round(x) doesn't work properly in Python 3, hence int(x+0.5)
        best_s += [values[i]]*n
    return supp(best_s, c)

def get_penalty3(pref): #v.3
    penalty = pref*(max_m-m_rec)+(1-pref)*abs(m_rec-m_cand) #v.3.1
    return abs(penalty)

def get_penalty4(pref): #v.4
    if m_cand == 0 or m_cand == 1:
        return 1
    else:
        return pref*(1-m_rec)+(1-pref)*max(1-m_rec/m_cand,(m_rec-m_cand)/(1-m_cand))
        #return pref*(max_m-m_rec)+(1-pref)*max(1-m_rec/(m_cand-min_m),(m_rec-m_cand)/(max_m-m_cand))
        
def get_penalty5(pref): #v.5
    return pref*(1-(m_rec-min_m)/(max_m-min_m))+(1-pref)*abs(m_rec-m_cand) \
                /max(m_cand-min_m,max_m-m_cand)
    
def get_penalty6(pref): #v.6
    return pref*(1-(m_rec-min_m)/(max_m-min_m))+(1-pref)*(1-supp(imp_rec, 
                imp_cand)/max_supp(imp_cand, len(imp_rec)))

def get_penalty7(pref, subset=None): #v.7
    if subset is None:
        subset = imp_rec
    m_rec = np.mean(subset)
    m_star = m_cand+pref*(max_m-m_cand)
    #print('m_star={}'.format(m_star))
    delta_min, delta_max = get_delta_bounds(m_star, subset.size)
    delta = abs(m_star-m_rec)
    #print ('delta-delta_min = {}'.format(delta-delta_min))
    #print ('delta_max-delta_min = {}'.format(delta_max-delta_min+1e-17))
    #penalty = (delta-delta_min)/(delta_max-delta_min+1e-17)
    #print ('penalty = {0:.2f}'.format(penalty))
    return (delta-delta_min)/(delta_max-delta_min+1e-17)
'''   
def get_best_subset_mean(c, x, size): # find the mean of subset of c of given size that is closest to x
    subsets = list(itertools.combinations(c, size))
    sums = np.asarray(list(map(sum, subsets)))
    best_subset = subsets[np.argmin(np.abs(sums/size-x))]
    return np.mean(best_subset)
'''

def combinations(subset_size):
    comb_iterator = unicomb.combinations(imp_cand, subset_size)
    combinations = np.stack(comb_iterator)
    means = np.asarray(list(map(np.mean, combinations)))
    return combinations, means

def get_delta_bounds(m_star, subset_size): # helper for v.7
    if subset_size != imp_rec.size:
        _, means = combinations(subset_size)
        deltas = np.abs(means-m_star)
    else:
        deltas = np.abs(subset_means-m_star)
    return np.min(deltas), np.max(deltas)

def get_cost(score_subset):
    w = min(tp.size, score_subset.size)/max(tp.size, score_subset.size)
    s = sum(score_subset-b)
    #print('w={}'.format(w))
    #print('s={}'.format(s))
    return -w*s

def optimal_subset():
    min_cost = sys.float_info.max
    best_subset = None
    for k in range(1, imp_cand.size+1): 
        #subsets_iterator = unicomb.combinations(imp_cand, k)
        imp_subsets_ids = itertools.combinations(range(scores_cand.size), k)
        for ids_tuple in imp_subsets_ids:
            imp_subset = np.asarray(imp_cand[list(ids_tuple)])
            epsilon = get_penalty7(pref_list[0], subset=imp_subset)
            #print ('subset {0}, epsilon = {1:.2f}, alpha-fair: {2}'
            #       .format(np.asarray(imp_cand[list(ids_tuple)]),epsilon,epsilon<=alpha))
            if epsilon <= alpha:
                scores_subset = np.asarray(scores_cand[list(ids_tuple)])
                cost = get_cost(scores_subset)
                if cost <= min_cost:
                    min_cost = cost
                    best_subset = scores_subset
    return best_subset

def generate_plot_data(imp_cand, imp_rec_size, pref):
    x = []
    y = []
    for i in np.argsort(subset_means):
        x.append(subset_means[i])
        y.append(get_penalty7(pref, subsets[i]))
    return x, y

def normalize(ranking, min_value, max_value):
    if min_value == max_value:
        return np.ones(ranking.size)
    else:
        return np.array([(x-min_value)/(max_value-min_value) for x in ranking])
    
def run(imp_cand, imp_rec, pref_list): 
    print('-----Input-----')
    print('imp_cand: ' + str(imp_cand))
    print('imp_rec: ' + str(imp_rec))
    
    #imp_cand = normalize(imp_cand, min_value, max_value)
    #imp_rec = normalize(imp_rec, min_value, max_value)
    
    #x_points = np.linspace(min_m, max_m, 500)
    plt.title('fairness penalty curve')
    plt.xlabel('imp_rec mean (m_rec)')
    plt.ylabel('penalty')

    for pref in pref_list:
        print('pref = %.2f\n' % pref)
        # v.1
        m_star = m_cand + pref * (max_m - m_cand)
        delta_min, delta_max = get_delta_bounds(m_star, imp_rec.size)
        error_upper_bound = max(m_star - min_m, max_m - m_star)
        e_v1 = abs(m_rec - m_star)
        e_norm = e_v1 / error_upper_bound
        f_norm = (1 - e_norm) * 100 # normalized fairness
        # v.3
        #e_v3 = m_rec - pref * max_m - (1 - pref) * np.mean(get_diff(imp_cand, imp_rec))
        e_v3 = get_penalty3(pref)
        e_v4 = get_penalty4(pref)
        e_v5 = get_penalty5(pref)
        e_v6 = get_penalty6(pref)
        e_v7 = get_penalty7(pref)
        # utilitarian equation
        #pref1 = pref / 2.0 + 0.5
        e_util = pref * (max_m - m_rec) + (1 - pref) * abs(m_cand - m_rec)
        #plt.plot(x, abs(x - m_star), label='pref='+str(pref)+', f_abs='+f"{f_abs:.2f}"+'%')
        #plt.plot(m_rec, e_abs, 'or')
        #plt.plot(x_points, [get_fairness_penalty(imp_cand, imp_rec, pref, x) for x in x_points], label='pref='+str(pref)+', f_abs='+f"{f_v3:.2f}"+'%') # plot error curve
        
        x, y = generate_plot_data(imp_cand, imp_rec.size, pref)
        plt.plot(x, y, label='pref='+str(pref)+', fairness='+f"{(1-e_v7)*100:.2f}"+'%') # plot error curve
        plt.plot(m_rec, e_v7, 'or') # plot rec.set mean point
        plt.legend()
        
        #e_b = pref * r_max / 2 + pref * r_avg / 2 + r_avg / 2 + r_max / 2 - r_l
        print('-----Quantities-----')
        print('m_cand = %.2f' % m_cand)
        print('m_rec = %.2f' % m_rec)
        print('m_diff = %.2f' % np.mean(get_diff(imp_cand, imp_rec)))
        print('min_m = %.2f' % min_m)
        print('max_m = %.2f' % max_m)
        print('m_star = %.2f' % m_star)
        print('delta_s = %.2f' % e_v1)
        print('delta_min = %.2f' % delta_min)
        print('delta_max = %.2f' % delta_max)
        #print('subsets: {}'.format(subsets))
        #print('subset means: {}'.format(subset_means))
        print('-----Results-----')
        #print('error_upper_bound = %.2f' % error_upper_bound)
        print('e_v1 = %.2f (fairness = %.2f%%)' % (e_v1, (1-e_v1)*100))
        #print('e_norm = %.2f (f_norm = %.2f)' % (e_norm, f_norm))
        print('e_util = %.2f (fairness = %.2f%%)' % (e_util, (1-e_util)*100))
        print('e_v3 = %.2f (fairness = %.2f%%)' % (e_v3, (1-e_v3)*100))
        print('e_v4 = %.2f (fairness = %.2f%%)' % (e_v4, (1-e_v4)*100))
        print('e_v5 = %.2f (fairness = %.2f%%)' % (e_v5, (1-e_v5)*100))
        print('e_v6 = %.2f (fairness = %.2f%%)' % (e_v6, (1-e_v6)*100))
        print('e_v7 = %.2f (fairness = %.2f%%)' % (e_v7, (1-e_v7)*100))
        print('---------\n')
        plt.legend()
    plt.show()
    #return r_avg, r_min, r_max, r_star, r_l, e_d
    
def prob_same_rel():
    same_rel_bb = 0.05**2+0.14**2+0.53**2+0.02**2+0.04**2+0.02**2+0.01**2+0.01**2+0.18**2
    same_rel_ww = 0.19**2+0.29**2+0.03**2+0.19**2+0.02**2+0.01**2+0.02**2+0.24**2+0.01**2
    same_rel_aa = 0.06**2+0.17**2+0.11**2+0.16**2+0.01**2+0.05**2+0.01**2+0.06**2+0.03**2+0.31**2+0.01**2
    same_rel_wa = 0.19*0.17+0.11*0.29+0.01*0.03+0.05*0.19+0.01*0.02+0.31*0.24+0.01*0.01
    same_rel_ba = 0.05*0.17+0.14*0.11+0.04*0.05+0.02*0.06+0.18*0.31
    same_rel_bw = 0.05*0.19+0.14*0.29+0.04*0.19+0.02*0.01+0.24*0.18
    same_rel_bl = 0.05*0.48+0.14*0.19+0.01*0.53+0.02*0.02+0.04*0.05+0.01*0.01+0.18*0.20
    same_rel_bo = 0.05*0.10+0.14*0.34+0.53*0.01+0.02*0.01+0.04*0.13+0.02*0.01+0.01*0.01+0.01*0.04+0.18*0.27

    print("prob_same_rel(black_black) = %.2f" % (same_rel_bb*100)+"%")
    print("prob_same_rel(white_white) = %.2f" % (same_rel_ww*100)+"%")
    print("prob_same_rel(asian_asian) = %.2f" % (same_rel_aa*100)+"%")
    print("prob_same_rel(white_asian) = %.2f" % (same_rel_wa*100)+"%")
    print("prob_same_rel(black_white) = %.2f" % (same_rel_bw*100)+"%")
    print("prob_same_rel(black_asian) = %.2f" % (same_rel_ba*100)+"%")
    print("prob_same_rel(black_latino) = %.2f" % (same_rel_bl*100)+"%")
    print("prob_same_rel(black_other) = %.2f" % (same_rel_bo*100)+"%")
    
    race_ids = range(5)
    probs = [same_rel_bb, same_rel_bw, same_rel_ba, same_rel_bl, same_rel_bo]
    races = ('black','white','asian','latino','other')
    bars = plt.bar(race_ids, probs, alpha=0.5)
    plt.xticks(race_ids, races)
    plt.ylabel('prob_same_rel')
    plt.ylim(0.0, 1.0)
    plt.title('prob_same_rel for race=black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2., 1.05*height, '%.2f' % height, ha='center', va='bottom') 
    
def main():
    #start = time.time()
    run(imp_cand, imp_rec, pref_list)
    #print('time: {}'.format(time.time()-start))
    #prob_same_rel()

    '''
    result = supp(imp_rec, imp_cand)
    print('supp = {}'.format(result)) 
    #best_s = max_supp(imp_cand, len(imp_rec))
    #print('max_supp: {}'.format(best_s))
    confidence = math.log(supp(imp_rec, imp_cand))
    confidence = 1/confidence if confidence > 0 else 1
    print('confidence: {0:.2f}'.format(confidence))

    x = comb(10, 3)
    #print(x)
    '''

    '''
    #test = np.array([0.4,0.3,0.2,0.05,0.01,-0.01,-0.1,-0.2,-0.3,-0.4])
    #full_set = np.array([0.9,0.9,0.9,0.6,0.51,0.49,0.40,0.30,0.20,0.10])
    full_set = np.array([1,0,0,1,1,0,1,0,0,0])
    subset = np.array([1])
    print('ratio: {}'.format(np.mean(full_set[:7])*np.sum(full_set[:7])))
    print('ratio: {}'.format(np.mean(full_set[:6])*np.sum(full_set[:6])))
    print('ratio: {}'.format(np.mean(full_set[:5])*np.sum(full_set[:5])))
    print('ratio: {}'.format(np.mean(full_set[:4])*np.sum(full_set[:4])))
    print('ratio: {}'.format(np.mean(full_set[:3])*np.sum(full_set[:3])))
    print('ratio: {}'.format(np.mean(full_set[:2])*np.sum(full_set[:2])))
    print('ratio: {}'.format(np.mean(full_set[:1])*np.sum(full_set[:1])))
    
    print('sum: {}'.format(np.sum(full_set[:6])))
    '''
    
    #c = get_cost(0.5)
    #print(c)
    
    start = time.time()
    print(optimal_subset())
    print('time: {}'.format(time.time()-start))

    
    '''
    scores = np.array([0.4,0.3,0.2,0.05])
    imps = np.array([0,1,1,0])
    comb_indices = list(itertools.combinations(range(4),3))
    for ind_tuple in comb_indices:
        ind_list = list(ind_tuple)
        print (imps[ind_list])
    '''
if __name__ == '__main__': main()