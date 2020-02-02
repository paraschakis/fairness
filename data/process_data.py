import numpy as np
from data import transformations
from data import tuning
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

def transform(X_train_df, y_train_df, X_test_df, y_test_df, clf,
              rebalancing=True, feature_selection=True):

    # impute missing data
    imputation_pipe = transformations.make_imputation_pipe()
    X_train = imputation_pipe.fit_transform(X_train_df)
    X_test = imputation_pipe.transform(X_test_df)
    
    # some feature entineering
    X_train['age_diff'] = abs(X_train['age'] - X_train['age_o'])
    X_test['age_diff'] = abs(X_test['age'] - X_test['age_o'])
    X_train_df['age_diff'] = X_train['age_diff']
    X_test_df['age_diff'] = X_test['age_diff']
    X_train['same_field'] = (X_train['field'] == X_train['field_o']).astype(int)
    X_test['same_field'] = (X_test['field'] == X_test['field_o']).astype(int)
    X_train.drop(columns=['field', 'field_o'], inplace=True)
    X_test.drop(columns=['field', 'field_o'], inplace=True)
    X_train['imp_race_diff'] = abs(X_train['imp_same_race'] - X_train['imp_same_race_o'])
    X_test['imp_race_diff'] = abs(X_test['imp_same_race'] - X_test['imp_same_race_o'])
    X_train['imp_rel_diff'] = abs(X_train['imp_same_rel'] - X_train['imp_same_rel_o'])
    X_test['imp_rel_diff'] = abs(X_test['imp_same_rel'] - X_test['imp_same_rel_o'])
    
    # encode data
    encoding_pipe = transformations.make_encoding_pipe(X_train)
    X_train = encoding_pipe.fit_transform(X_train)
    X_test = encoding_pipe.transform(X_test)
    
    y_train = y_train_df.values
    y_test = y_test_df.values
        
    # Rebalance data (if needed)
    if rebalancing:
        print('Rebalancing...')
        rebalance_pipe = imbPipeline([('rebalance', None),
                                      ('classify', clf)])
        grid = tuning.rebalance_grid(rebalance_pipe)
        grid.fit(X_train, y_train)
        sampler = grid.best_estimator_.named_steps['rebalance']
        if sampler:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        print('best params: {}'.format(grid.best_params_))
    
    # Perform feature selection (if needed)
    if feature_selection:
        print('Reducing dimensionality...')
        reduce_dim_pipe = Pipeline([('reduce_dim', None),
                                      ('classify', clf)])
        grid = tuning.reduce_dim_grid(reduce_dim_pipe)
        grid.fit(X_train, y_train)
        feature_selector = grid.best_estimator_.named_steps['reduce_dim']
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_test = feature_selector.transform(X_test)
        print('best params: {}'.format(grid.best_params_))
    return X_train, y_train, X_test, y_test

def tune_threshold(X_train, y_train, clf):
    # tune threshold
    print('Tuning threshold...')
    folds = 10
    skf = StratifiedKFold(n_splits=folds)
    thresholds = []
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_f, X_test_f = X_train[train_index], X_train[test_index]
        y_train_f, y_test_f = y_train[train_index], y_train[test_index]
        clf.fit(X_train_f, y_train_f)
        y_proba = clf.predict_proba(X_test_f)
        best_threshold = 0.5
        best_score = roc_auc_score(y_test_f, y_proba[:, 1] > best_threshold)
        for threshold in np.linspace(0.1, 0.9, num=100):
            score = roc_auc_score(y_test_f, y_proba[:, 1] > threshold)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        thresholds.append(best_threshold)
    threshold = sum(thresholds) / folds
    print('optimal threshold: {}'.format(threshold))
    return threshold
