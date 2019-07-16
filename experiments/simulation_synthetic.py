import sys
import os
# enable imports from sibling directories
sys.path.insert(0, os.path.abspath('..'))

import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import numpy as np
from scipy import stats

from config import *
import metrics.rank as metrics
from algorithms.exhaustive import Exhaustive
from algorithms.knapsack import Knapsack
from algorithms.tabu import Tabu
from metrics.epsilon import Epsilon
from metrics.epsilon_ss import Epsilon_ss


# Settings
np.random.seed = 7
runs = 100
start_size = 1
end_size = 21
alpha = 0
b = 0.5


def sample(size):
    scores = np.array(list(map(lambda x: round(x, 2), np.random.rand(size))))
    scores = np.sort(scores)
    if max(scores) < b:  # filter out bad samples
        return sample(size)
#    probsame = np.random.randint(0, 2, size)  # binary
    probsame = np.array(list(map(lambda x: round(x,2), np.random.rand(size))))  # continuous
    # pref = np.random.randint(0, 11)/10
    pref = np.random.rand()
    np.set_printoptions(precision=2)
    return scores, probsame, pref


def print_df(df, title):
    print('\n---------------')
    print(title)
    print('---------------')


def plot(df, title, ylim=None, logy=False):
    print_df(df, title)
    i = len(df.columns) - 1
    plt.figure(figsize=(6, 5))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': FONT_SIZE,
                         'xtick.labelsize': TICK_LABEL_SIZE + 2,
                         'ytick.labelsize': TICK_LABEL_SIZE + 2,
                         'axes.titlesize': TITLE_SIZE + 4,
                         'axes.labelsize': AXES_LABEL_SIZE + 4,                        
                         'figure.autolayout': True,                         
                         })
    size = list(range(start_size, end_size + 1))
    for column in df:
        means = np.mean(df[column].tolist(), axis=1)
        print('means({0}): {1}'.format(column, means))
        stderr = stats.sem(df[column].tolist(), axis=1)
        plt.plot(size, means, linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                 marker=MARKERS[i], color=PALETTE1[i], label=column)
        plt.fill_between(size, means - stderr, means + stderr, color=PALETTE1[i], alpha=0.10)
        i -= 1
    if logy:
        plt.yscale('log')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())  # remove scientific notation
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc=(0.3, 0.21), prop={'size': LEGEND_SIZE+4})
    plt.xlabel('Size')
    plt.title(title)
    plt.grid(alpha=GRID_ALPHA)
    plt.savefig(OUTPUT_PATH + title + f'.{SAVE_FORMAT}', dpi=DPI, format=SAVE_FORMAT)
    plt.show()


def run():
#    algos = {Exhaustive: {},
#             Tabu: {'tabu_size': 1},
#             Knapsack: {'prec': 2}}
    algos = {Exhaustive: {},
             Tabu: {'tabu_size': 1},
             Knapsack: {'prec': 2}}
    sim_type = 'Jaccard'  # 'Dice'
    f1_data = []
    sim_data = []
    opt_rate_data = []
    runtime_data = []
    for size in range(start_size, end_size + 1):
        f1_per_run = np.zeros(shape=(runs, len(algos) + 1))  # +1 because of lower bound
        sim_per_run = np.zeros(shape=(runs, len(algos)))
        runtimes_per_run = np.zeros(shape=(runs, len(algos)-1))
        for i in range(runs):
            print("\rsize {0} run {1}... ".format(size, i + 1), end="", flush=True)
            scores, probsame, pref = sample(size)
            outputs, runtimes = generate_outputs(algos, scores, probsame, pref)
            rec_set = scores[np.argmax(scores >= b):]
            f1_per_run[i] = calc_f1(rec_set, outputs)
            sim_per_run[i] = calc_sim(sim_type, outputs)
            runtimes_per_run[i] = runtimes
        f1_data.append(tuple(i for i in f1_per_run.T))
        sim_data.append(tuple(i for i in sim_per_run.T))
        runtime_data.append(tuple(i for i in runtimes_per_run.T))
        opt_rate = np.where(sim_per_run < 1.0, 0, 1)
        opt_rate_data.append(tuple(i for i in opt_rate.T))

    plot(pd.DataFrame(f1_data,
                      columns=('Upper bound', 'Lower bound', 'Tabu search', 'Knapsack',)), 'F1 loss')
    plot(pd.DataFrame(sim_data,
                      columns=('Lower bound', 'Tabu search', 'Knapsack')), sim_type + ' similarity')
    plot(pd.DataFrame(runtime_data,
                      columns=('Tabu search', 'Knapsack')), 'Running time (s)')
    plot(pd.DataFrame(opt_rate_data,
                      columns=('Lower bound', 'Tabu search', 'Knapsack')), 'Global optimum rate')

def generate_outputs(algos, scores, probsame, pref):
    outputs = []
    runtimes = []
    columns = []
    for alg, args in algos.items():
        algorithm = alg(pref, scores, probsame, **args)
        start = time.time()
        output = algorithm.run(alpha)
        runtime = time.time() - start
        columns.append(alg.__name__)
        if isinstance(output, tuple):  # needed for lower bound
            outputs.extend(output)
            columns.append('Lower bound')
        else:
            runtimes.append(runtime) # measure time for knapsack and tabu
            outputs.append(output)
    df_outputs = pd.DataFrame([outputs], columns=columns)
    return df_outputs, runtimes


def calc_f1(rec_set, outputs):
    results = []
    for alg in outputs:
        results.append(metrics.f1(rec_set, outputs[alg][0]))
    return results


def calc_sim(sim_type, outputs):
    results = []
    ref = outputs['Exhaustive'][0]
    outputs.drop('Exhaustive', axis=1, inplace=True)
    if sim_type == 'Jaccard':
        similarity = metrics.jaccard_similarity
    if sim_type == 'Dice':
        similarity = metrics.dice_similarity
    for alg in outputs:
        results.append(similarity(ref, outputs[alg][0]))
    return results


def run_two():
    result = []
    alg1 = 'exhaustive'
    alg2 = 'knapsack'
    runtime = {alg1: [], alg2: []}
    for size in range(start_size, end_size + 1):
        local_result = []
        time_alg1 = []
        time_alg2 = []
        for i in range(runs):
            print("\rsize {0} run {1}... ".format(size, i + 1), end="", flush=True)
            scores, probsame, pref = sample(size, b)
            rec_set = scores[np.argmax(scores >= b):]

            alg_exhaustive = Exhaustive(pref, scores, probsame, b, rel_func=lambda w, s: w * s)
            alg_knapsack = Knapsack(pref, scores, probsame, b, prec=2, rel_func=lambda w, s: w * s)
            # alg_tabu = Tabu(pref, scores, probsame, b, rel_func=lambda w,s: w*s)

            start = time.time()
            rec_alg1, _ = alg_exhaustive.run(pref, alpha)
            time_alg1.append((time.time() - start) * 1000)

            start = time.time()
            rec_alg2 = alg_knapsack.run(pref, alpha)
            # rec_alg2, _ = alg_exhaustive.run1(pref, alpha)
            # rec_alg2 = alg_tabu.run(alpha)
            time_alg2.append((time.time() - start) * 1000)

            try:  # if two arrays are not equal in size, exception will be raised
                local_result.append(int(np.allclose(rec_alg1, rec_alg2)))  # array_equal doens't work well with floats
            except:
                local_result.append(0)

            '''
            print('\nscores: {}'.format(scores))
            print('probsame: {}'.format(probsame))
            print('pref: {}'.format(pref))
            print('exhaustive solution: {}'.format(rec_alg1))
            print('knapsack solution: {}'.format(rec_alg2))
            break
            '''
        # print(local_result)
        result.append(np.mean(local_result))
        runtime[alg1].append(np.mean(time_alg1))
        runtime[alg2].append(np.mean(time_alg2))
    print(result)
    print(f'runtime {alg1}: {runtime[alg1]}')
    print(f'runtime {alg2}: {runtime[alg2]}')


def run_epsilon():
    matches = []
    runtime_old = []  # old (exhaustive) epsilon calculation
    runtime_new = []  # new (subset sum) epsilon calculation

    for size in range(start_size, end_size + 1):
        t_old = []
        t_new = []
        m = []
        for run in range(runs):
            print("\rsize {0} run {1}... ".format(size, run + 1), end="", flush=True)
            scores, probsame, pref = sample(size, b)
            rec_ids = np.nonzero(scores >= b)
            probsame_rec = probsame[rec_ids]

            start = time.time()
            e = Epsilon(pref, probsame, probsame_rec)
            score = round(e.score(), 2)
            t_old.append(time.time() - start)

            start = time.time()
            e1 = Epsilon_ss(pref, probsame, probsame_rec, prec_numeric=13, prec_knapsack=2)
            score1 = round(e1.score(), 2)
            t_new.append(time.time() - start)

            m.append(int(score == score1))

            '''
            if not score == score1:
                print('\n ============== ')
                print('\nepsilon: {}'.format(score))
                print('epsilon1: {}'.format(score1))
                print('scores: {}'.format(scores))
                print('probsame: {}'.format(probsame))
                print('pref: {}'.format(pref))
            '''

        runtime_old.append(np.mean(t_old))
        runtime_new.append(np.mean(t_new))
        matches.append(np.mean(m))

    print('\nruntime_old:\n {}'.format(runtime_old))
    print('runtime_new:\n {}'.format(runtime_new))
    print('match ratio:\n {}'.format(matches))


# run_two()
# test_epsilon()

run()


'''
def plot(df, title, ylim=None, logy=False):
    print_df(df, title)
    marker = itertools.cycle(('o','x','+', ',', '.'))
    for column in df:
        if column == 'Size':
            size = df[column]
        else:
            means = np.mean(df[column].tolist(), axis=1)
            print('means({0}): {1}'.format(column, means))
            stderr = stats.sem(df[column].tolist(), axis=1)
            plt.plot(size, means, marker = next(marker), label=column)
            plt.fill_between(size, means-stderr, means+stderr, alpha=0.1)
    if logy:
        plt.yscale('log')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter()) # remove scientific notation
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc='best')
    plt.xlabel('Size')
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.show()
'''

'''
def run_full():
    sim_data = []
    glob_opt_data = []
    time_data = []
    f1_data = []
    for size in range(start_size, end_size+1):
        sim_greedy = np.zeros(runs)
        sim_lower_bound = np.zeros(runs)
        time_greedy = np.zeros(runs)
        time_exhaustive = np.zeros(runs)
        f1_greedy = np.zeros(runs)
        f1_exhaustive = np.zeros(runs)
        f1_lower_bound = np.zeros(runs)
        for run in range(runs):
            print("\rsize {0} run {1}... ".format(size, run+1), end="", flush=True)
            scores, probsame, pref = sample(size, b)
            rec_set = scores[np.argmax(scores >= b):]
            alg_exhaustive = Exhaustive(scores, probsame, b, rel_func=lambda w,s: w*s)
            start = time.time()
            rec_exhaustive, lower_bound = alg_exhaustive.run(pref, alpha)
            time_exhaustive[run] = (time.time()-start)*1000
            #print('exhaustive: {}'.format(rec_exhaustive))
            #alg_greedy = Greedy(scores, probsame, b, rel_func=lambda w,s: w*s)
            alg_greedy = Tabu(pref, scores, probsame, b, rel_func=lambda w,s: w*s)
            start = time.time()
            rec_greedy = alg_greedy.run(alpha)
            time_greedy[run] = (time.time()-start)*1000
            #print('greedy: {}'.format(rec_greedy))
            f1_greedy[run] = metrics.rank.f1(rec_set, rec_greedy)
            f1_exhaustive[run] = metrics.rank.f1(rec_set, rec_exhaustive)
            f1_lower_bound[run] = metrics.rank.f1(rec_set, lower_bound)
            sim_greedy[run] = metrics.rank.dice_similarity(rec_exhaustive, rec_greedy)
            sim_lower_bound[run] = metrics.rank.dice_similarity(rec_exhaustive, lower_bound)
        print("done")
        glob_opt_greedy = np.where(sim_greedy < 1.0, 0, 1)
        #sim_data.append((size, np.mean(sim_greedy)))
        sim_data.append((size, sim_greedy, sim_lower_bound))
        glob_opt_data.append((size, glob_opt_greedy))
        time_data.append((size, time_greedy, time_exhaustive))
        f1_data.append((size, f1_greedy, f1_exhaustive, f1_lower_bound))
    
    plot(pd.DataFrame(sim_data, columns=('Size', 'Greedy', 'Lower bound')), 'Dice similarity')
    plot(pd.DataFrame(glob_opt_data, columns=('Size', 'Greedy')), 'Global optimum rate')
    plot(pd.DataFrame(time_data, columns=('Size', 'Greedy', 'Exhaustive')), 
        'Running time (ms)', logy=False)
    plot(pd.DataFrame(f1_data, columns=('Size', 'Greedy', 'Exhaustive', 'Lower bound')), 'F1')
'''
