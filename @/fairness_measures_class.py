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

class Metrics:
    
    def __init__(self, cand_set, rec_set, pref_list):
        print('-----Input-----')
        print('cand_set: ' + str(cand_set))
        print('rec_set: ' + str(rec_set))
        
        #cand_set = normalize(cand_set, min_value, max_value)
        #rec_set = normalize(rec_set, min_value, max_value)
        
        self.cand_set = cand_set
        self.rec_set = rec_set
        self.pref_list = pref_list
        self.m_cand = np.mean(cand_set)
        self.m_rec = np.mean(rec_set)
        self.min_m, self.max_m = self.get_bounds()
        subsets = itertools.combinations(cand_set, rec_set.size)
        self.subsets = np.asarray(list(dict.fromkeys(subsets))) # remove duplicates
        self.means_of_subsets = np.asarray(list(map(np.mean, self.subsets)))
    
    def get_bounds(self):
        min_m = np.mean(np.sort(self.cand_set)[:self.rec_set.size])
        max_m = np.mean(-np.sort(-self.cand_set)[:self.rec_set.size])
        return min_m, max_m

    def get_diff(self, minuend, subtrahend):
        for element in subtrahend:
            i = np.where(minuend == element)[0][0] # get index of first match
            minuend = np.delete(minuend, i)
        return minuend

    def supp(self, s, c): # calculates 'combinatorial support' supp(s|c)
        values_s, counts_s = np.unique(s, return_counts=True)
        result = 1
        for i in range(len(values_s)):
            count_c = (c == values_s[i]).sum()
            result *= comb(count_c, counts_s[i])
        return result
    
    def max_supp(self, c, size): # calculates maximum support for a given set and a target size
        best_s = [] # set that maximizes supp(s|c)
        values, counts = np.unique(c, return_counts=True)
        for i in range(len(values)):
            r = counts[i] / len(c)
            n = min(int(r*size+0.5), size-len(best_s)) # round(x) doesn't work properly in Python 3, hence int(x+0.5)
            best_s += [values[i]]*n
        return self.supp(best_s, c)
    
    def get_delta_bounds(self, x): # calculate min and max distances of subsets from the given mean
        deltas = np.abs(self.means_of_subsets-x)
        return np.min(deltas), np.max(deltas)

    def generate_plot_data(self, pref):
        x = []
        y = []
        for i in np.argsort(self.means_of_subsets):
            x.append(self.means_of_subsets[i])
            y.append(self.get_penalty7(pref, self.subsets[i]))
        return x, y

    def normalize(self, ranking, min_value, max_value):
        if min_value == max_value:
            return np.ones(ranking.size)
        else:
            return np.array([(x-min_value)/(max_value-min_value) for x in ranking])

    def get_penalty3(self, pref): #v.3
        penalty = pref*(self.max_m-self.m_rec)+(1-pref)*abs(self.m_rec-self.m_cand) #v.3.1
        return abs(penalty)
    
    def get_penalty4(self, pref): #v.4
        if self.m_cand == 0 or self.m_cand == 1:
            return 1
        else:
            return pref*(1-self.m_rec)+(1-pref)*max(1-self.m_rec/self.m_cand,
                        (self.m_rec-self.m_cand)/(1-self.m_cand))
        
    def get_penalty5(self, pref): #v.5
        return pref*(1-(self.m_rec-self.min_m)/(self.max_m-self.min_m))+ \
                        (1-pref)*abs(self.m_rec-self.m_cand)/max(self.m_cand-self.min_m,self.max_m-self.m_cand)
        
    def get_penalty6(self, pref): #v.6
        left_term = pref*(1-(self.m_rec-self.min_m)/(self.max_m-self.min_m))
        right_term = (1-pref)*(1-self.supp(self.rec_set, 
                     self.cand_set)/self.max_supp(self.cand_set, len(self.rec_set)))
        return left_term+right_term
                    
    def get_penalty7(self, pref, rec_set=None): #v.7
        if rec_set is None:
            rec_set = self.rec_set
        m_star = self.m_cand+pref*(self.max_m-self.m_cand)
        delta_min, delta_max = self.get_delta_bounds(m_star)
        delta = abs(m_star-np.mean(rec_set))
        return (delta-delta_min)/(delta_max-delta_min)
    
    def calculate(self): 
        #x_points = np.linspace(min_m, max_m, 500)
        plt.title('fairness penalty curve')
        plt.xlabel('rec_set mean (m_rec)')
        plt.ylabel('penalty')
    
        for pref in self.pref_list:
            print('pref = %.2f\n' % pref)
            # v.1
            m_star = self.m_cand + pref * (self.max_m - self.m_cand)
            error_upper_bound = max(m_star - self.min_m, self.max_m - m_star)
            e_v1 = abs(self.m_rec - m_star)
            e_norm = e_v1 / error_upper_bound
            f_norm = (1 - e_norm) * 100 # normalized fairness
            # v.3
            #e_v3 = m_rec - pref * max_m - (1 - pref) * np.mean(get_diff(cand_set, rec_set))
            e_v3 = self.get_penalty3(pref)
            e_v4 = self.get_penalty4(pref)
            e_v5 = self.get_penalty5(pref)
            e_v6 = self.get_penalty6(pref)
            e_v7 = self.get_penalty7(pref)
            # utilitarian equation
            #pref1 = pref / 2.0 + 0.5
            e_util = pref * (self.max_m - self.m_rec) + (1 - pref) * abs(self.m_cand - self.m_rec)
            #plt.plot(x, abs(x - m_star), label='pref='+str(pref)+', f_abs='+f"{f_abs:.2f}"+'%')
            #plt.plot(m_rec, e_abs, 'or')
            #plt.plot(x_points, [get_fairness_penalty(cand_set, rec_set, pref, x) for x in x_points], label='pref='+str(pref)+', f_abs='+f"{f_v3:.2f}"+'%') # plot error curve
            
            x, y = self.generate_plot_data(pref)
            plt.plot(x, y, label='pref='+str(pref)+', fairness='+f"{(1-e_v7)*100:.2f}"+'%') # plot error curve
            plt.plot(self.m_rec, e_v7, 'or') # plot rec.set mean point
            plt.legend()
            
            #e_b = pref * r_max / 2 + pref * r_avg / 2 + r_avg / 2 + r_max / 2 - r_l
            print('-----Metrics-----')
            print('m_cand = %.2f' % self.m_cand)
            print('m_rec = %.2f' % self.m_rec)
            print('m_diff = %.2f' % np.mean(self.get_diff(self.cand_set, self.rec_set)))
            print('min_m = %.2f' % self.min_m)
            print('max_m = %.2f' % self.max_m)
            print('m_star = %.2f' % m_star)
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
    
    def prob_same_rel(self):
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
    #cand_set = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    #cand_set = np.array([0,0,0,1,1,1,1,1,1,1])
    #cand_set = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    #rec_set = np.array([0,1,0,0,0])
    #rec_set = np.array([0,1,1,1])
    #cand_set = np.array([0.33,0.27,0.25,0.2,0.19,0.15,0.1,0.09,0.08,0.04])
    #cand_set = np.array([0.3,0.3,0.3,0.3,0.2,0.2,0.4,0.4,0.4,0.4])
    cand_set = np.array([0.1,0.2,0.3,0.3,0.3,0.5,0.7,0.7,0.8,0.8])
    rec_set = np.array([0.1,0.3,0.7])
    #rec_set = np.array([0.3,0.3,0.3,0.3])
    #rec_set = np.array([0.4,0.4,0.2,0.2])
    #rec_set = np.array([0.33,0.33,0.33,0.33,0.33])
    pref = [0.0, 0.5, 1.0]

    metrics = Metrics(cand_set, rec_set, pref)
    metrics.calculate()
    #generate_plot_data(cand_set, rec_set.size, pref[0])
    #prob_same_rel()
    
    
    #subsets = list(itertools.combinations(cand_set,3))
    #means = np.asarray(list(map(np.mean, subsets)))
    #best_subset = subsets[np.argmin(np.abs(sums-np.mean(cand_set)*3))]
    #result = np.mean(cand_set)-np.min(np.abs(sums/3-np.mean(cand_set)))
    #print ("best subset: {}".format(best_subset))
    #print ("best subset's mean: {}".format(result))
    #delta_min = np.min(np.abs(means-np.mean(cand_set)))
    #delta_max = np.max(np.abs(means-np.mean(cand_set)))
    #print ("delta min: {}".format(delta_min))
    #print ("delta max: {}".format(delta_max))

    
    result = metrics.supp(rec_set, cand_set)
    print('supp = {}'.format(result)) 
    #best_s = max_supp(cand_set, len(rec_set))
    #print('max_supp: {}'.format(best_s))
    confidence = math.log(metrics.supp(rec_set, cand_set))
    confidence = 1/confidence if confidence > 0 else 1
    print('confidence: {0:.2f}'.format(confidence))

    x = comb(10, 3)
    #print(x)
    
    nCk1 = comb(6,5)
    nCk2 = comb(4,1)
    nCk3 = comb(200,5)
    #print("nCk1=%d" % nCk1)
    #print("nCk2=%d" % nCk2)
    #print("nCk3=%d" % nCk3)
    #print(nCk1*nCk2/nCk3)

    
if __name__ == '__main__': main()

''' 

Problem with utilitarian equation (e_b): 
    (1) for balanced cand_set and pref=0, it gives different errors for the case when recommendations
only contains all 1s, and when they only contain all 0s. e_abs gives 0.5 in both cases (since pref=0 means indifference).
    (2) for cand_set = [0 0 0 0 0 0 1 1 1 1], rec_set = [1 0 0 0 1] and pref = 0.0, it gives e_util = 0.2. It should be 0.0, 
    since the means are the same.

Problem with e_v1: when cand_set = [0 0 0 0 0 0 0 1 1 1], rec_set = [1 1 0 0 1] and pref = 0.0. Turns out that e_abs = error_upper_bound,
but the fairness is 70% (too optimistic for maximum error?). Maybe it's not a problem.

Problem with e_v4: when cand_set = [0 0 0 0 0 1 1 1 1 1], rec_set = [0 1 1]. For all preference values, it gives e = 0.33.
Problem with e_v3: when cand_set = [0 0 0 0 0 1 1 1 1 1], rec_set = [0 1 1 1]. For all preference values, it gives e = 0.25. 
'''



'''
def generate_plot_data(cand_set, rec_set_size, pref):
    x = []
    y = []
    means = []
    sorted_cand_set = -np.sort(-cand_set)
    for i in range(cand_set.size - rec_set_size + 1):
        for j in range(i + 1, cand_set.size - rec_set_size + 2):
            current_set = np.append(np.array(sorted_cand_set[i]), np.array(sorted_cand_set[j:j+rec_set_size-1]))
            m_current_set = np.mean(current_set)
            if not m_current_set in means:
                x.append(np.mean(current_set))
                y.append(get_fairness_penalty(cand_set, current_set, pref))
                means.append(m_current_set)
    #print (x)
    #print (y)
    return x, y



    
    '''