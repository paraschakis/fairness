import pandas as pd
import numpy as np
from data import load_data
import seaborn as sns
import matplotlib.pyplot as plt
from config import *


# draw bar chart with racial distribution of partners
# separately for all candidates, and for picked ones (i.e. where decision=1)
def plot_preference_vs_decision(df):
    
    # set plot settings
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': FONT_FAMILY,
                         'xtick.labelsize': TICK_LABEL_SIZE + 2,
                         'ytick.labelsize': TICK_LABEL_SIZE + 2,
                         'axes.titlesize': TITLE_SIZE,
                         'axes.labelsize': AXES_LABEL_SIZE + 2,
                         'figure.autolayout': True,
                         'axes.grid': False
                         })
    plt.rcParams.update({})

    same = df['same_race'] == 1
    same_vs_imp_same = [(df[same].imp_same_race == i).sum() for i in range(1, 11)]
    diff_vs_imp_same = [(df[~same].imp_same_race == i).sum() for i in range(1, 11)]

    same_vs_imp_same_dec = [((df[same].imp_same_race == i) &
                             (df[same].dec == 1)).sum() for i in range(1, 11)]
    diff_vs_imp_same_dec = [((df[~same].imp_same_race == i) &
                             (df[~same].dec == 1)).sum() for i in range(1, 11)]

    # bar plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    x = np.linspace(1, 10, 10)
    ax.bar(x - 0.21, same_vs_imp_same, width=0.4, color=PALETTE3[2],
           log=True, label='same-race candidates')
    ax.bar(x + 0.21, diff_vs_imp_same, width=0.4, color=PALETTE3[1],
           log=True, label='different-race candidates')
    ax.bar(x - 0.21, same_vs_imp_same_dec, width=0.4, color=PALETTE1[1],
           log=True, hatch='////', label='same-race picks')
    ax.bar(x + 0.21, diff_vs_imp_same_dec, width=0.4, color=PALETTE1[2],
           log=True, hatch='////', label='different-race picks')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2,  
    #          prop={'size': LEGEND_SIZE + 2})
    ax.legend(loc='upper right', ncol=2,  
          prop={'size': LEGEND_SIZE})
    ax.set_xticks(x)
    ax.set_xlabel('same-race importance')
    ax.set_title('Racial distribution of dating partners')
    ax.set_facecolor('1.0')
    plt.savefig(OUTPUT_PATH + 'partner_distribution' + f'.{SAVE_FORMAT}', dpi=DPI,
                format=SAVE_FORMAT)
    plt.show()


# draw heat maps for imp_same_race vs. like
# seprately for same-race and diff.-race partners
def plot_preference_vs_like(df):
    
    # set plot settings
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': FONT_FAMILY,
                         'xtick.labelsize': TICK_LABEL_SIZE - 4,
                         'ytick.labelsize': TICK_LABEL_SIZE - 4,
                         'axes.titlesize': TITLE_SIZE - 6,
                         'axes.labelsize': AXES_LABEL_SIZE - 4,
                         })
    
    df.dropna(subset=['like'], inplace=True)
    same = df['same_race'] == 1

    # round some fractional likes and separate data by same_race
    df['like'] = round(df['like'])
    imp_same_likes_same = df[same][['imp_same_race', 'like']]
    imp_same_likes_diff = df[~same][['imp_same_race', 'like']]

    # create multiindex with distribution of likes per each imp_same_rame group
    likes_distr_same = imp_same_likes_same.groupby('imp_same_race')['like'] \
        .value_counts(normalize=True)
    likes_distr_diff = imp_same_likes_diff.groupby('imp_same_race')['like'] \
        .value_counts(normalize=True)

    # convert multiindex to 2d-numpy (imp_same_race=0 is dropped for being rare)
    likes_matrix_same = likes_distr_same.unstack(fill_value=0).to_numpy().T[1:, 1:]
    likes_matrix_diff = likes_distr_diff.unstack(fill_value=0).to_numpy().T[1:, 1:]

    # plot heat maps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5), sharey=True)
    ax0 = sns.heatmap(likes_matrix_same, ax=ax[0], square=True)
    ax1 = sns.heatmap(likes_matrix_diff, ax=ax[1], square=True)
    ax0.invert_yaxis()
    tick_labels = list(range(1, 11))
    ax0.set_xticklabels(tick_labels)
    ax0.set_yticklabels(tick_labels, rotation=0)
    ax1.set_xticklabels(tick_labels)
    ax0.set(xlabel='same-race importance',
            ylabel='like',
            title='same-race partners')
    ax1.set(xlabel='same-race importance', title='diff.-race partners')
    plt.savefig(OUTPUT_PATH + 'like_heat_maps' + f'.{SAVE_FORMAT}', dpi=DPI,
                format=SAVE_FORMAT)
    plt.show()
    
def print_wave_info(df):
    waves = len(df.index.unique(2))
    print('\nnumber of waves: {}'.format(waves))
    df_reset = df.reset_index(level=[0,1,2])
    min_participants = df_reset.groupby(['wave'])['iid'].nunique().min()
    max_participants = df_reset.groupby(['wave'])['iid'].nunique().max()
    min_pairs = df_reset['wave'].value_counts().min()
    max_pairs = df_reset['wave'].value_counts().max()
    print('min participants per wave: {}'.format(min_participants))
    print('max participants per wave: {}'.format(max_participants))
    print('min pairs per wave: {}'.format(min_pairs))
    print('max pairs per wave: {}'.format(max_pairs))
    
def print_population(df):
    total_number = len(df.index.unique(0))
    males = len(df[df['gender'] == 'Male'].index.unique(0))
    females = len(df[df['gender'] == 'Female'].index.unique(0))
    print('\nnumber of participants: {}'.format(total_number))
    print('number of males: {0} ({1:.2f}%)'.format(
            males, males / total_number * 100))
    print('number of females: {0} ({1:.2f}%)'.format(
            females, females / total_number * 100))
    print('number of pairs: {}'.format(len(df)))

def print_racial_info(df):
    same = df['same_race'] == 1
    print('\nnumber of same-race pairs: {0} ({1:.2f}%)'.format(
        len(df[same]), len(df[same]) / len(df) * 100))
    print('number of different-race pairs: {0} ({1:.2f}%)'.format(
        len(df[~same]), len(df[~same]) / len(df) * 100))

def print_outcomes(df):
    matches = len(df[df['match'] == 1])
    match_ratio = matches / len(df)
    decisions = len(df[df['dec'] == 1])
    dec_ratio = decisions / len(df)
    print('\nmatches: {0} ({1:.2f}%)'.format(
            matches, match_ratio * 100))
    print('positive decisions: {0} ({1:.2f}%)'.format(
            decisions, dec_ratio * 100))
    pos_decision_males = len(df[(df['dec'] == 1) & 
                                (df['gender'] == 'Male')])
    pos_decision_females = len(df[(df['dec'] == 1) & 
                                  (df['gender'] == 'Female')])
    print('decisions in males: {0} ({1:.2f}%)'.format(
        pos_decision_males, pos_decision_males / decisions * 100))
    print('decisions in females: {0} ({1:.2f}%)'.format(
        pos_decision_females, pos_decision_females / decisions * 100))



def main():
    
    df = load_data.processed()

    print('\n======= DATASET STATISTICS =======')
    
    print_wave_info(df)
    print_population(df)
    print_racial_info(df)
    print_outcomes(df)

    print('\n======= PREFERENCE CONSISTENCY PLOTS =======')

    plot_preference_vs_decision(df)
    plot_preference_vs_like(df)


if __name__ == '__main__':
    main()
