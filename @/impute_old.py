import numpy as np

# =============================================================================
# impute missing values
# =============================================================================

def impute_missing(df):

    # age, age_o: impute based on the mode of age difference
    df_m = df[df['gender'] == 'Male']
    mode_diff_m = (df_m['age'] - df_m['age_o']).mode()[0].astype(int)
    mode_diff_f = -mode_diff_m
    df['age'] = np.where((df['gender'] == 'Male') & (df['age'].isnull()),
                         df['age_o'] + mode_diff_m, df['age'])
    df['age'] = np.where((df.gender == 'Female') & (df['age'].isnull()),
                         df['age_o'] + mode_diff_f, df['age'])
    df['age_o'] = np.where((df['gender'] == 'Male') & (df['age_o'].isnull()),
                           df['age'] + mode_diff_f, df['age_o'])
    df['age_o'] = np.where((df['gender'] == 'Female') & (df['age_o'].isnull()),
                           df['age'] + mode_diff_m, df['age_o'])

    # go_date: impute based on mode
    # all missing go_date entries have go_out='Twice a week'
    # these two fields are expected to be correlated
    mask = df['go_out'] == 'Twice a week'
    mode = df[mask].go_date.mode()[0]
    df.go_date.where(df.go_date.notnull(), other=mode, inplace=True)

    # 'rate yourself' features: impute based on median, mode, or mean
    def impute_rate_yourself(features):
        for f in features:
            df[f].where(df[f].notnull(), other=df[f].median(), inplace=True)
    impute_rate_yourself(['attr', 'sinc', 'fun', 'int', 'amb'])

    # pf_fun, pf_amb, and pf_sha
    # impute 0s because all 'points' went to other pf_ attributes
    cols = ['pf_fun', 'pf_amb', 'pf_sha']
    df[cols] = df[cols].fillna(0)

    # pf_o_att, pf_o_fun, pf_o_amb, pf_o_sin, pf_o_sha, pf_o_int
    # one person with these attributes missing is a 26 y.o. white girl (id=28)
    # for each attribute, get the mean from similar people (white females 26)
    # the calculation is done over the corresponding pf_ attributes
    def impute_pf_o(pf_o_features):
        df_sim = df.loc[(df['race'] == 'European/Caucasian-American') &
                        (df['age'] == 26)]
        for f in pf_o_features:
            pf_feature = f.replace('o_', '')
            df[f].where(df['pid'] != 28,
                        other=round(df_sim[pf_feature].mean()), inplace=True)
    impute_pf_o(['pf_o_att', 'pf_o_fun', 'pf_o_amb', 
                 'pf_o_sin', 'pf_o_sha', 'pf_o_int'])
    # the rest of missing pf_o features should be imputed with 0s because
    # all points went to other pf_o attributes
    cols = ['pf_o_fun', 'pf_o_amb', 'pf_o_sha']
    df[cols] = df[cols].fillna(0)

    # exp_happy: impute by mode
    df.exp_happy.where(df.exp_happy.notnull(), other=df.exp_happy.mode()[0],
                       inplace=True)

    # int_corr: impute by mean because it is continuous
    df.int_corr.where(df.int_corr.notnull(), other=df.int_corr.mean(),
                      inplace=True)


def print_missing(df):
    null_cols = df.columns[df.isnull().any()]
    if not null_cols.empty:
        print('---- columns with missing values ---- ')
        print(df[null_cols].isnull().sum())
        print('------------------------------------- ')
