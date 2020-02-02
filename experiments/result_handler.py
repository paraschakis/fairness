from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from metrics.epsilon import Epsilon
from algorithms.knapsack import Knapsack
from algorithms.tabu import Tabu
from data import load_data
from config import *
from math import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

PATH = 'dumps/y=match (feat_sel tuning) FINAL/'
CLF_LIST = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'GaussianNB']

class ClassifierResult:
    
    def __init__(self, name, df, make_fair=True, reranker=None, alpha=0):
#        self.num_users = len(next(iter(result_dict.values())))
        self.make_fair = make_fair
        self.alpha = alpha  # fairness threshold (used only when make_fair=True)
        self.name = name
        self.df = df
        self.f1_micro = 0   # average over concatenated predictions
        self.f1_micro_fair = 0  # same as above but after enforcing fairness
        self.f1_macro = 0   # average over predictions per user
        self.f1_macro_fair = 0  # same as above but after enforcing fairness
        self.auc_micro = 0
        self.auc_micro_fair = 0
        self.coverage = 0   # ratio of non-empty predictions
        self.informativeness = 0   # ratio of predictions that are not 'all 1s'
        self.informativeness_after = 0
        self.fairness = 0
        self.fairness_lb = 0 # lower bound (fairness of informative recommendations)
        self.fairness_after = 0
        self.reranker = 'Knapsack' if reranker==None else reranker
        self._process_df(df)
    
    def _process_df(self, df):
        threshold = df['threshold'].iloc[0]
        y_true = []
        y_proba = []
        y_pred_fair = []
        empty_recs = 0
        uninformative_recs = 0
        uninformative_recs_after = 0
        f1_sum = 0
        f1_sum_after = 0
        f1_count = 0
        f1_count_after = 0
        fairness_sum = 0
        fairness_sum_lb = 0
        fairness_sum_after = 0
        fairness_count = 0
        fairness_count_lb = 0
        print('Re-ranker: {}'.format(self.reranker))
        for row in df.itertuples():  # row corresponds to a user
            print("\rAnalyzing user {0}/{1}... ".format(f1_count, len(df)), 
                  end="", flush=True)
            y_true.extend(row.y_true)
            y_proba.extend(row.y_proba)
            mask = np.array(row.y_proba) >= threshold
            same_r_rec = row.same_r[mask]
            if same_r_rec.size > 0: # recommendations exist
                # calculate fairness
                e = Epsilon(row.imp_same_r, row.same_r, same_r_rec)
                fairness_sum += e.score(as_fairness=True)
                fairness_count += 1                
                if same_r_rec.size == len(row.y_true):
                    uninformative_recs += 1 # recommended everyone
                else: # compute 'informative' fairness (i.e. lower bound)
                    fairness_sum_lb += e.score(as_fairness=True)
                    fairness_count_lb += 1
                # calculate macro-f1
                y_pred = mask.astype(int)
                f1_sum += f1_score(row.y_true, y_pred)
                f1_count += 1
                if self.make_fair:
                    if round(e.score(), 2) <= round(self.alpha, 2): # already fair
                        fair_solution = y_pred # original solution
                    else: # generate fair solution
                        # call re-ranker
                        algorithm = globals()[self.reranker]\
                        (row.imp_same_r,
                         row.y_proba,
                         row.same_r,
                         b=threshold,
                         # prec=3, # only for Knapsack for continuous attributes
                         return_ids=True,
                         rel_func=lambda w, s: w*s)
                        ids = algorithm.run(alpha=self.alpha)
                        fair_solution = np.zeros(len(row.y_true)).astype(int)
                        fair_solution[ids] = 1
                        e = Epsilon(row.imp_same_r, row.same_r, row.same_r[ids])
                    y_pred_fair.extend(fair_solution)
                    f1_sum_after += f1_score(row.y_true, fair_solution)
                    f1_count_after += 1
                    if sum(fair_solution) == len(row.y_true):
                        uninformative_recs_after += 1
                fairness_sum_after += e.score(as_fairness=True)
            else:
                # if no recommendations are given, assign all zeros to y_pred_fair
                y_pred_fair.extend(np.zeros(len(row.y_true)).astype(int))
                uninformative_recs += 1
                uninformative_recs_after += 1
                if sum(row.y_true) > 0: # no recommendations, but match exists
                    # count as empty only if a match exists (otherwise empty is ok)
                    empty_recs += 1
                    f1_count += 1 # corresponds to f1 = 0
                    f1_count_after += 1
                else: # no recommendations for no matches -> correct!
                    f1_sum += 1
                    f1_sum_after += 1
                    f1_count += 1
                    f1_count_after += 1
        n_users = len(df)
        self.coverage = 100 * (n_users - empty_recs) / n_users
        self.informativeness =  100 * (n_users - uninformative_recs) / n_users
        y_pred = (np.array(y_proba) >= threshold).astype(int)
        self.f1_micro = f1_score(y_true, y_pred)
        self.f1_macro = f1_sum / f1_count
        self.auc_micro = roc_auc_score(y_true, y_pred)
        try:
            self.fairness = fairness_sum / fairness_count
            self.fairness_lb = fairness_sum_lb / fairness_count_lb
        except ZeroDivisionError:
            print('no recommendations are given')
        if self.make_fair:
            try:
                self.f1_micro_fair = f1_score(y_true, y_pred_fair)
                self.f1_macro_fair = f1_sum_after / f1_count_after
                self.auc_micro_fair = roc_auc_score(y_true, y_pred_fair)
                self.fairness_after = fairness_sum_after / fairness_count
            except ZeroDivisionError:
                print('no recommendations are given')
            self.informativeness_after = (n_users - uninformative_recs_after) / n_users * 100
        
    def __str__(self):
        name = self.name.split('Classifier')[0]
        header = (
            f'\n\n'
            f'*****************************\n'
            f'{name}\n'
            f'*****************************\n'
        )
        sample = (
            f'-------- sample --------\n'
            f'number of users: {len(self.df)}\n'
            f'waves analyzed: {self.df.wave.unique()}\n'
        )
        main_result = (
            f'-------- result --------\n'
            f'f1: {self.f1_micro:.4f}\n'
            f'f1 (macro): {self.f1_macro:.4f}\n'
            f'auc: {self.auc_micro:.4f}\n'
            f'coverage: {self.coverage:.2f}%\n'
            f'informativeness: {self.informativeness:.2f}%\n'
            f'fairness: {self.fairness:.2f}% (lower bound: {self.fairness_lb:.2f}%)\n'
        )
        fair_result = (
            f'----- {self.reranker} -----\n'
            f'f1: {self.f1_micro_fair:.4f}\n'
            f'f1 (macro): {self.f1_macro_fair:.4f}\n'
            f'auc: {self.auc_micro_fair:.4f}\n'
            f'fairness: {self.fairness_after:.2f}%\n'
            f'alpha: {self.alpha:.1f}\n'
            f'informativeness: {self.informativeness_after:.2f}%\n'
        )
        output = header + sample + main_result
        if self.make_fair:   
            output += fair_result
        return output
    
    
def plot_unfair_vs_fair(clf_results, save_figure=False):
    # set plot settings
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': FONT_FAMILY,
                         'xtick.labelsize': TICK_LABEL_SIZE-4,
                         'ytick.labelsize': TICK_LABEL_SIZE-4,
                         'axes.titlesize': TITLE_SIZE-4,
                         'axes.labelsize': AXES_LABEL_SIZE-4,
                         })
    
    x = np.array(range(1, len(clf_results)+1))
    f1_unfair = [result.f1_micro for result in clf_results]
    f1_fair = [result.f1_micro_fair for result in clf_results]
    names = [result.name.split('Classifier')[0] for result in clf_results]
    
    # bar plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.bar(x - 0.21, f1_unfair, width=0.4, color=PALETTE3[2], label='original')
    ax.bar(x + 0.21, f1_fair, width=0.4, color=PALETTE3[1], hatch='////',
           label='fair')
    ticks_and_labels = plt.xticks(x, names)
    for i, label in enumerate(ticks_and_labels[1]):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    plt.legend(loc='best', prop={'size': LEGEND_SIZE-4})
    ax.set_ylabel('f1', rotation=0)
    ax.set_title('fairness-aware vs. fairness-agnostic')
    if save_figure:
        plt.savefig(OUTPUT_PATH + 'f1' + f'.{SAVE_FORMAT}', dpi=DPI,
                format=SAVE_FORMAT)
    plt.show()
    

def process_results(output_dict, save_dir=False, make_fair=True,
                    reranker=None, alpha=0):
    # save to file
    if save_dir:
        for name, df in output_dict.items():
            df.to_pickle(save_dir + name + '.df')
    
    # process dataframes of each classifier
    clf_results = []
    for name, df in output_dict.items():
        print('Classifier: {}'.format(name))
        clf_result = ClassifierResult(name, df, make_fair, reranker, alpha)
        print(clf_result)
        clf_results.append(clf_result)
    return clf_results
#    plot_unfair_vs_fair(clf_results, save_figure=True)


def load_results(path=PATH, clf_list=CLF_LIST):
    outputs = dict()
    filenames = clf_list if clf_list else listdir(path)
    for filename in filenames:
        try:
            name, _ = filename.split('.')
        except ValueError: # when clf_list is provided
            name = filename
            filename += '.df'
            if not isfile(join(path, filename)):
                continue # if it is a directory
        outputs[name] = pd.read_pickle(join(path, filename))
    return outputs


# replaces same_r and imp_same_r in original data with religion equivalents
def load_results_religion(path_religion, reranker=None, clf_list=CLF_LIST):
    results = load_results(clf_list=clf_list)
    results_religion = load_results(path_religion, clf_list)
    for clf, df in results_religion.items():
        results[clf]['same_r'] = df['same_r']
        results[clf]['imp_same_r'] = df['imp_same_r']
    return results


def plot_alphas(alphas, results, reranker, save_figure=False):
    # set plot settings
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': FONT_FAMILY,
                         'xtick.labelsize': TICK_LABEL_SIZE+2,
                         'ytick.labelsize': TICK_LABEL_SIZE+2,
                         'axes.titlesize': TITLE_SIZE,
                         'axes.labelsize': AXES_LABEL_SIZE+2,
                         'figure.autolayout': True,
                         })
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.set_xlabel('fairness (%)', rotation=0)
    ax.set_ylabel('F1 score', rotation=90)
    ax.set_title(f'F1 vs. fairness ({reranker})')

    x = list(map(lambda x: round((1-x)*100), alphas))

    i = len(results)-1
    for name, data in results.items():
        if name == 'RandomForestClassifier':
            name = 'Random Forest'
        if name == 'LogisticRegression':
            name = 'Logistic Regression'        
        if name == 'GaussianNB':
            name = 'Naive Bayes'
        if name == 'XGBClassifier':
            name = 'XGBoost'
        plt.plot(x, data, label=name, linewidth=LINE_WIDTH, 
                 markersize=MARKER_SIZE, marker=MARKERS[i], color=PALETTE1[i])
        i -= 1    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, 
               prop={'size': LEGEND_SIZE+3})
    plt.grid(alpha=GRID_ALPHA)
    if save_figure:
        plt.savefig(OUTPUT_PATH + f'f1_vs_fairness_{reranker}_religion' + f'.{SAVE_FORMAT}', dpi=DPI,
                format=SAVE_FORMAT)    
    plt.show()

# experiment: test re-rankers for various values of alpha
def experiment_alphas(alphas, outputs, reranker, save_figure=True):
    results = defaultdict(list)
    for alpha in alphas:
        print('alpha: {}'.format(alpha))
        clf_results = process_results(outputs, make_fair=True,
                                      reranker=reranker, alpha=alpha)
        for r in clf_results:
            results[r.name].append(r.f1_macro_fair)
    plot_alphas(alphas, results, reranker, save_figure=save_figure)


if __name__ == '__main__':
    # results = load_results()
    results = load_results_religion('dumps/religion/')
    # process_results(results, reranker='Tabu')
    experiment_alphas([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                      results,
                      'Tabu',
                      save_figure=False)