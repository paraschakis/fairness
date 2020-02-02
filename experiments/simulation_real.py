import numpy as np
import pandas as pd
from data import load_data
from data import transformations
from data import tuning
from data import process_data
from experiments import result_handler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
import sys
import warnings

INPUT_FILE = '../data/processed/speed_dating_religion.csv'

if not sys.warnoptions:
    warnings.simplefilter("ignore")

seed = 11
np.random.seed = seed

tune_threshold = True
calibration = False
tune_parameters = False
rebalancing = False
feature_selection = True

results_path = 'dumps/'

interests = ['sports','tvsports','exercise','dining','museums','art',
                'hiking','gaming','clubbing','reading','tv','theater','movies',
                'concerts','music','shopping','yoga','sports_o','tvsports_o',
                'exercise_o','dining_o','museums_o','art_o',
                'hiking_o','gaming_o','clubbing_o','reading_o','tv_o','theater_o',
                'movies_o','concerts_o','music_o','shopping_o','yoga_o']
self_eval = ['attr','sinc','fun','int','amb','attr_o','sinc_o','fun_o','int_o',
             'amb_o']
imp = ['imp_same_race', 'imp_same_rel', 'imp_same_race_o', 'imp_same_rel_o']
exp_happy = ['exp_happy', 'exp_happy_o']
field = ['field','field_o']
go = ['goal','go_date','go_out','goal_o','go_date_o','go_out_o']
nominal_columns = ['race', 'race_o', 'goal', 'goal_o'] 
to_drop = interests + exp_happy

X, y = load_data.X_y_split(file=INPUT_FILE, cols_to_drop=to_drop)

p = sum(y)
n = len(y) - p

clf_random = DummyClassifier(strategy='stratified', random_state=seed)
clf_LR = LogisticRegression(solver='liblinear', class_weight='balanced',
                            random_state=seed)
clf_NB = GaussianNB()
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=3, 
                                class_weight='balanced', random_state=seed, 
                                criterion='gini', n_jobs=-1)
clf_XGB = XGBClassifier(random_state=seed, n_estimators=300, scale_pos_weight=n/p)

# define classifiers
classifiers = [clf_LR, clf_RF, clf_XGB, clf_NB]

# initialize results
results = dict()
for clf in classifiers:
    result_df = pd.DataFrame(columns=['iid', 'wave', 'y_proba', 'y_true', 
                               'same_r', 'imp_same_r', 'threshold'], dtype=object)
    result_df = result_df.set_index(['iid'])
    results[clf.__class__.__name__] = result_df

for wave in range(1, 22):
    print(f'\n******* WAVE {wave} *******\n')

    # create train-test splits for the current wave
    idx = pd.IndexSlice
    train_waves = [i for i in range(1, 22) if i != wave]
    X_train_df = X.loc[idx[:, :, train_waves], :]
    y_train_df = y.loc[idx[:, :, train_waves]]
    X_test_df = X.loc[idx[:, :, wave], :]
    y_test_df = y.loc[idx[:, :, wave]]
    
    
    for clf in classifiers:
        
        print(f'\nRunning {clf.__class__.__name__}...\n')
             
        # transform data
        X_train, y_train, X_test, y_test = process_data.transform(
                X_train_df, y_train_df, X_test_df, y_test_df, clf,
                rebalancing=rebalancing, feature_selection=feature_selection)
        
        name = clf.__class__.__name__     
        threshold = 0.5
        
        # tune threshold and perform grid search
        if name is not 'DummyClassifier':
            if calibration:
                clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
            if tune_parameters:
                clf = tuning.classifier_grid(clf)            
        clf.fit(X_train, y_train)
               
        try: # if it is a grid search object
            print('best params: {}'.format(clf.best_params_))
            
            if tune_threshold:
                threshold = process_data.tune_threshold(X_train, y_train, clf.best_estimator_)
        except AttributeError:
            if tune_threshold:
                threshold = process_data.tune_threshold(X_train, y_train, clf)
        
        y_proba = clf.predict_proba(X_test)[:, 1]
        if name == 'DummyClassifier':
            y_proba = np.random.rand(len(y_test))

        id_index = y_test_df.index.get_level_values('iid')
        for user_id in id_index.unique():
            user_slice = id_index.get_loc(user_id)
            results[name].at[user_id, 'y_proba'] = y_proba[user_slice]
            results[name].at[user_id, 'y_true'] = y_test[user_slice]
            results[name].at[user_id, 'wave'] = wave
            results[name].at[user_id, 'same_r'] = \
                        (X_test_df[user_slice]['same_religion']).to_numpy()
            results[name].at[user_id, 'imp_same_r'] = \
                 X_test_df[user_slice]['imp_same_rel'][0].astype(int)
            results[name].at[user_id,'threshold'] = threshold

result_handler.process_results(results, save_dir=results_path, make_fair=False)
