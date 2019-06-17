import numpy as np
import pandas as pd
from data import load_data
from data import transformations
from data import tuning
from data import process_data
from experiments import result_handler
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import scikitplot as skplt
import os
import csv
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

seed = 11
np.random.seed(seed)

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

raw = load_data.raw()
X, y = load_data.X_y_split(cols_to_drop=to_drop)

p = sum(y)
n = len(y) - p

clf_random = DummyClassifier(strategy='stratified', random_state=seed)
#clf_all1 = DummyClassifier(strategy='constant', constant=1, random_state=seed)
clf_LR = LogisticRegression(solver='liblinear', class_weight='balanced', 
                            random_state=seed)
clf_KNN = KNeighborsClassifier(n_jobs=-1)
clf_QDA = QuadraticDiscriminantAnalysis()
clf_NB = GaussianNB()
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=3, 
                                class_weight='balanced', random_state=seed, 
                                criterion='gini', n_jobs=-1)
clf_XGB = XGBClassifier(random_state=seed, n_estimators=300, scale_pos_weight=n/p)
clf_GB = GradientBoostingClassifier(random_state=seed)
clf_MLP = MLPClassifier(max_iter=100, random_state=seed)

# define classifiers
classifiers = [clf_LR, clf_RF, clf_XGB, clf_NB]
#classifiers = [clf_XGB]

# initialize results
results = dict()
columns = ['iid', 'gender', 'race', 'imp_same_race', 'imp_same_rel', 
           'goal', 'go_date', 'go_out',
           'pf_att', 'pf_sin', 'pf_int', 'pf_fun', 'pf_amb', 'pf_sha',
           'attr', 'sinc', 'fun', 'int', 'amb',
           'mean_age_diff_rec', 'mean_age_diff_cand']
for clf in classifiers:
#    result_df = pd.DataFrame(columns=['iid', 'wave', 'y_proba', 'y_true', 
#                               'same_r', 'imp_same_r', 'threshold'], dtype=object)
    result_df = pd.DataFrame(columns=columns, dtype=object)    
    result_df = result_df.set_index(['iid'])    
    results[clf.__class__.__name__] = result_df
    
auc = dict.fromkeys(classifiers, 0)
wave_counter = 0

for wave in range(1, 22):
    print(f'\n******* WAVE {wave} *******\n')
    wave_counter += 1
    
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
            
        auc[clf] += roc_auc_score(y_test, y_proba)

        id_index = y_test_df.index.get_level_values('iid')
        for user_id in id_index.unique():
            user_slice = id_index.get_loc(user_id)
            age_diff_cand = X_test_df[user_slice]['age_diff']
            mean_age_diff_cand = age_diff_cand.mean()
            predictions = y_proba[user_slice]
            ids = np.argsort(predictions)[-3:]
            age_diff_rec = X_test_df[user_slice]['age_diff'][ids]
            mean_age_diff_rec = age_diff_rec.mean()
            scaler = MinMaxScaler()
            age_diff_cand_array = age_diff_cand.to_numpy().reshape(-1, 1)
            scaler.fit(age_diff_cand_array)
            mean_age_diff_cand = scaler.transform(age_diff_cand_array).mean()
            mean_age_diff_rec = scaler.transform(mean_age_diff_rec.reshape(1,-1)).mean()
            
            to_drop = {'mean_age_diff_rec', 'mean_age_diff_cand', 'iid'}
            columns = [e for e in columns if e not in to_drop]
            slice_df = X_test_df[columns][user_slice]
            results[name].at[user_id, columns] = slice_df.iloc[0] # copy all user data to results
            results[name].at[user_id, 'mean_age_diff_cand'] = round(mean_age_diff_cand, 2)
            results[name].at[user_id, 'mean_age_diff_rec'] = round(mean_age_diff_rec, 2)
print()

for clf, score in auc.items():
    print('auc({0}): {1:.2f}'.format(clf.__class__.__name__, score / wave_counter))

path = 'dumps/'
for clf, output in results.items():
    output.to_csv(path + clf + '.csv')