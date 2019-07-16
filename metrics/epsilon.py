# -*- coding: utf-8 -*-
"""
Calculates epsilon measure, taking values from 0 (best) to 1 (worst).
The numerical precision has a significant impact on the accuracy.
For prec > 3, exhaustive and knapsack seem to correlate well.
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt
from config import *

np.random.seed = 7

class Epsilon:
    
    """
    pref: preference of user (e.g. importance of the same race)
    cand: vector of candidates expressed in the attribute of interest (e.g. same race)
    rec: vector of recommendations expressed in the attribute of interest (e.g. same race)
    """

    def __init__(self, pref, probsame_cand, probsame_rec, prec=13):
        self.pref = pref
        self.cand = probsame_cand
        self.rec = probsame_rec
        self.prec = prec
        self.m_cand = np.mean(self.cand)
        sorted_cand = np.sort(self.cand)
        # self.m_min = np.mean(sorted_cand[:rec.size])
        # self.m_max = np.mean(sorted_cand[-rec.size:])
        self.m_min = round(np.mean(sorted_cand[:self.rec.size]), self.prec)
        self.m_max = round(np.mean(sorted_cand[-self.rec.size:]), self.prec)
        self.means = None
        self.subsets = None
        self.info = dict()

    # returns lower and upper bounds on subset means defined by alpha
    def alpha_bounds(self, alpha):
        m_star = self.m_star()
        delta_min, delta_max, _, _ = self._delta_bounds(m_star)
        delta_alpha = alpha * (delta_max - delta_min) + delta_min
        m_lower = max(m_star - delta_alpha, self.m_min)
        m_upper = min(m_star + delta_alpha, self.m_max)
        return round(m_lower, self.prec), round(m_upper, self.prec)

    # returns subsets whose means fall in the interval defined by alpha
    def alpha_subsets(self, alpha):
        m_lower, m_upper = self.alpha_bounds(alpha)
        return [self.subsets[i] for i in range(len(self.subsets))
                if m_lower <= self.means[i] <= m_upper]

    def _delta_bounds(self, m_star=None):
        if m_star is None:
            m_star = self.m_star()
        if self.means is None:
            self._generate_combinations()
        deltas = abs(self.means - m_star)
        min_index = np.argmin(deltas)
        max_index = np.argmax(deltas)
        self.delta_min, self.delta_max = deltas[min_index], deltas[max_index]
        subset_min, subset_max = self.subsets[min_index], self.subsets[max_index]
        return round(self.delta_min, self.prec), round(self.delta_max, self.prec), subset_min, subset_max

    def _generate_combinations(self):
        self.subsets = np.array(list(set(itertools.combinations(self.cand, self.rec.size))))
        self.means = np.array(list(map(np.mean, self.subsets)))
        self.means = np.asarray([round(x, self.prec) for x in self.means])
        # return self.subsets, self.means

    def _generate_plot_data(self, pref):
        x = []
        y = []
        for i in np.argsort(self.means):
            x.append(self.means[i])
            y.append(self.score(pref=pref, rec=self.subsets[i]))
        return x, y

    def m_star(self, pref=None):
        if pref is None:
            pref = self.pref
        return round(self.m_cand + pref * (self.m_max - self.m_cand), self.prec)
        # return self.m_cand+pref*(self.m_max-self.m_cand)

    def visualize(self, prefs=None):
        if prefs is None:
            prefs = [self.pref]
        if self.subsets is None:
            self._generate_combinations()

        plt.figure(figsize=(11, 6))
        plt.rcParams.update({'font.family': FONT_FAMILY, 
                     'font.size': FONT_SIZE,
                     'xtick.labelsize': TICK_LABEL_SIZE + 4,
                     'ytick.labelsize': TICK_LABEL_SIZE + 4,
                     'axes.titlesize': TITLE_SIZE + 2,
                     'axes.labelsize': AXES_LABEL_SIZE + 4,
                     })
        plt.rcParams.update({'figure.autolayout': True})
        # plt.rcParams.update({'text.usetex': True}) # enable latex mode
        plt.title('Unfairness penalty curves')
        plt.xlabel('solution mean (μ)')
        plt.ylabel('unfairness penalty (ε)')

        i = 0
        for pref in prefs:
            x, y = self._generate_plot_data(pref)
            score = self.score(pref)
            plt.plot(x, y, label='p=' + str(pref) + ', fairness=' + f"{(1 - score) * 100:.2f}" + '%',
                     linewidth=LINE_WIDTH+2, linestyle=LINE_STYLES[i % len(LINE_STYLES)], markersize=MARKER_SIZE,
                     color=PALETTE1[i % len(LINE_STYLES)])
            plt.plot(np.mean(self.rec), score, 'or', markersize=MARKER_SIZE+5)  # plot m_rec point
            print('\n--------------- [pref = {}] ---------------\n'.format(pref))
            print(self.info)
            i += 1

        #plt.legend(loc='upper center', prop={'size': LEGEND_SIZE + 2})
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': LEGEND_SIZE + 3})
        plt.grid(alpha=GRID_ALPHA)
        try:
            plt.savefig(OUTPUT_PATH + 'fairness_curve.' + SAVE_FORMAT, dpi=DPI, format=SAVE_FORMAT)
        except PermissionError:
            print('Permission error: could not write to file')
        plt.show()

    def score(self, pref=None, rec=None, as_fairness=False):
        if rec is None:
            rec = self.rec
        assert rec.size == self.rec.size, \
            "this object only allows rec. set of size {}".format(self.rec.size)
        m_rec = round(np.mean(rec), self.prec)
        m_star = self.m_star(pref)
        delta_min, delta_max, _, _ = self._delta_bounds(m_star)
        delta = abs(m_star - m_rec)
        if delta_min == delta_max:
            epsilon = 0
        else:
            epsilon = (delta - self.delta_min) / (self.delta_max - self.delta_min)
        self.info['m_star'] = m_star
        self.info['delta'] = delta
        self.info['epsilon'] = epsilon
        return (1 - epsilon) * 100 if as_fairness else epsilon
