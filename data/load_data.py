import pandas as pd
import os

INPUT_FILE = '../data/processed/speed_dating.csv'


# preprocessing steps safely applied to the entire dataset
def preprocess(df):
    # select columns from which missing values should be deleted
    # imputation to these columns is meaningless or harmful for analysis
    subset = ['pid', 'imp_same_race', 'race_o', 'int_corr']
    df.dropna(subset=subset, inplace=True)
    df.pid = df.pid.astype(int)

    # df['age_diff'] = df['age'] - df['age_o']

    # impute missing data that is not required to be done in a pipeline
    # impute.impute_missing(df)

    # pf_o_att, pf_o_fun, pf_o_amb, pf_o_sin, pf_o_sha, pf_o_int
    # one person with these attributes missing is a 26 y.o. white girl (id=28)
    # for each attribute, get the mean from similar people (white females 26)
    # the calculation is done over the corresponding pf_ attributes
    # (leads to insignificant data leakage, since it affects only 1 person)
    def impute_pf_o(pf_o_features):
        df_sim = df.loc[(df['race'] == 'European/Caucasian-American') &
                        (df['age'] == 26)]
        for f in pf_o_features:
            pf_feature = f.replace('o_', '')
            df[f].where(df['pid'] != 28,
                        other=round(df_sim[pf_feature].mean()), inplace=True)

    impute_pf_o(['pf_o_att', 'pf_o_fun', 'pf_o_amb',
                 'pf_o_sin', 'pf_o_sha', 'pf_o_int'])

    # impute in columns where NaN implies 0
    cols = ['pf_fun', 'pf_amb', 'pf_sha', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']
    df[cols] = df[cols].fillna(0)
    
    # add additional features of a partner
    cols_to_add = ['iid', 'imp_same_race', 'imp_same_rel', 'field', 'goal',
                   'go_date', 'go_out', 'sports', 'tvsports', 'exercise',
                   'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing',
                   'reading', 'tv', 'theater', 'movies', 'concerts', 'music',
                   'shopping', 'yoga', 'attr', 'sinc', 'fun', 'int', 'amb',
                   'exp_happy']
    df_unique_iid = df.drop_duplicates(subset='iid')[cols_to_add]
    df_merged = pd.merge(df, df_unique_iid, suffixes=('', '_o'),
                         left_on='pid', right_on='iid', how='left')
    df = df_merged.drop(columns='iid_o')
    # move outcomes to the end of dataframe
    df_like = df.pop('like')
    df_dec = df.pop('dec')
    df_match = df.pop('match')
    df['like'] = df_like
    df['dec'] = df_dec
    df['match'] = df_match

    # set multiindex
    df.set_index(['iid', 'pid', 'wave'], inplace=True)

    # set the correct data type for some columns
    # df['age'] = df['age'].astype('int64')
    # df['age_o'] = df['age_o'].astype('int64')
    return df

# =============================================================================
# data access methods
# =============================================================================

def raw(file=INPUT_FILE):
    return pd.read_csv(os.path.abspath(file))

# indexed, formatted, some missing values handled
def processed(file=INPUT_FILE):
    return preprocess(raw(file))


# processed and prepared for machine learning
def X_y_split(file=INPUT_FILE, cols_to_drop=None):
    df = processed(file)
    to_drop = ['like', 'dec']
    if cols_to_drop:
        to_drop += cols_to_drop
    X = df.drop(columns=to_drop)
    y = X.pop('match')
    return X, y

interests = ['sports','tvsports','exercise','dining','museums','art',
                'hiking','gaming','clubbing','reading','tv','theater','movies',
                'concerts','music','shopping','yoga']

self_eval = ['attr','sinc','fun','int','amb']

outcomes = ['like', 'dec']

# R = raw()
# P = processed()
# X, y = X_y_split(interests)








#def encode_features(df):
#    category_mapper = {'Almost never': 0,
#                       'Several times a year': 1,
#                       'Once a month': 2,
#                       'Twice a month': 3,
#                       'Once a week': 4,
#                       'Twice a week': 5,
#                       'Several times a week': 6}
#    df['go_date'] = df['go_date'].map(category_mapper)
#    df['go_out'] = df['go_out'].map(category_mapper)
#    return pd.get_dummies(df)