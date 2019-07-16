import numpy as np
import pandas as pd
from data import load_data
from data import transformations
from data import tuning
from data import process_data
from experiments import result_handler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
# import scikitplot as skplt
import os
import csv
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

# raw = load_data.raw(INPUT_FILE)
X, y = load_data.X_y_split(file=INPUT_FILE, cols_to_drop=to_drop)

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
#clf_XGB = XGBClassifier(random_state=seed)
clf_GB = GradientBoostingClassifier(random_state=seed)
clf_MLP = MLPClassifier(max_iter=100, random_state=seed)
#clf_NN = KerasClassifier(build_fn=None, epochs=50, batch_size=20, verbose=1)
#clf_SVN = LinearSVC(class_weight='balanced', random_state=seed, max_iter=5000)

# define classifiers
classifiers = [clf_LR, clf_RF, clf_XGB, clf_NB]
#classifiers = [clf_LR]

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
            # print()

path = 'dumps/' 
result_handler.process_results(results, save_dir=path, make_fair=False)



#            f1 = f1_score(y_test, y_proba > threshold)
#            print('------------')
#            print(f'f1 ({name}): {f1}')
#            print('num_pos (pred): {}'.format(sum(y_proba > threshold)))
#            print('------------')            
#            fpr, tpr, _ = roc_curve(y_test,  y_proba)
#            auc = roc_auc_score(y_test, y_proba)
#            plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#            plt.legend(loc=4)
#            plt.show()

        # Baselines
#        y_pred_random = np.random.randint(2, size=len(y_test))
#        y_pred_allpos = np.ones(len(y_test))
#        f1_random = f1_score(y_test, y_pred_random)
#        f1_allpos = f1_score(y_test, y_pred_allpos)
#        print('------------')
#        print('f1 (random): {}'.format(f1_random))
#        print('f1 (all positive): {}'.format(f1_allpos))
#        print('num_total: {}'.format(len(y_test)))
#        print('num_pos: {}'.format(sum(y_test)))
#        print('------------')
        

#        def create_NN():
#
#            # create model
#            model = Sequential()
#            model.add(Dense(100, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#        #        model.add(Dense(512, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#        #        model.add(Dense(256, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#        #        model.add(Dense(10, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#        #        model.add(Dropout(0.5))
#        #        model.add(Dense(100, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#        #        model.add(Dropout(0.5))
#            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#            # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
#            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#            return model
#        
#        if clf.__class__.__name__ is 'KerasClassifier':
#            clf = KerasClassifier(build_fn=create_NN, epochs=50, batch_size=20, verbose=1)
        
        
        

#            fop, mpv = calibration_curve(y_test, y_proba[:, 1], n_bins=10, normalize=True)
#            plt.plot([0, 1], [0, 1], linestyle='--')
#            plt.plot(mpv, fop, marker='.')
#            plt.show()


#        print_score(clf_random)
#        print_score(clf_all_pos) 
#    print_score(clf_LR)
#    print_score(clf_GXB)
#    print_score(clf_RF)
#    print_score(clf_LR)
#    print_score(clf_grid)
#    print_score(clf_MLP)
#    print_score(clf_NN)
#    print_score(clf_KNN)
#    print_score(clf_QDA)
#    print_score(clf_NB)

    #        opt_threshold, opt_f1 = optimize_threshold(y_proba)
    #        print(f'f1 optimized ({clf.__class__.__name__}): {opt_f1}')
    #        print(f'optimized threshold: {opt_threshold})')
    #        print('num_pos (pred after adjustment): {}'.format(sum(y_proba[:, 1] 
    #                                                   > opt_threshold)))

    #    pipe = Pipeline([('rebalance', None),
    #                        ('reduce_dim', None),
    #                        ('classify', clf),
    #                        ])
    
    #    model = grid_search.define_total_grid(pipe, X_train.shape[1])
    
   
    # over/under-sampling
    
#    sampler = SMOTE(kind = 'regular')
#    sampler = RandomOverSampler(random_state=0)
#    sampler = RandomUnderSampler(random_state=0)
#    sampler = InstanceHardnessThreshold(random_state=0,
#                               estimator=LogisticRegression(solver='liblinear', class_weight='balanced'))
#    sampler = EditedNearestNeighbours(random_state=0)
#    X_train, y_train = sampler.fit_resample(X_train, y_train)



#from imblearn.under_sampling import EditedNearestNeighbours
#from imblearn.under_sampling import RepeatedEditedNearestNeighbours
#from imblearn.under_sampling import AllKNN
#from imblearn.under_sampling import CondensedNearestNeighbour
#from imblearn.under_sampling import NeighbourhoodCleaningRule
#from imblearn.under_sampling import InstanceHardnessThreshold

    
    # feature selection

    
#    clf = DecisionTreeClassifier()
#    clf = LogisticRegression(solver='liblinear', class_weight='balanced')
#    trans = RFECV(clf, cv=3)
#    trans = SelectFromModel(clf, threshold='median')
#    trans = SelectKBest(mutual_info_classif, k=200)
#    trans = SelectFwe(alpha=0.01)
    
#    X_train = trans.fit_transform(X_train, y_train)
#    X_test = trans.transform(X_test)





    
#    scale_pos_weight = [0.5,1,2,3,4,5,6,7,8,9,10,11]
#    for i in scale_pos_weight:
#        clf = XGBClassifier(n_estimators=500, scale_pos_weight=i)
#        clf.fit(X_train, y_train)
#        y_proba = clf.predict_proba(X_test)
#        f1 = f1_score(y_test, y_proba[:, 1] > 0.5)
#        print(f'scale_pos_weight = {i}, f1 (xgb): {f1}')
        
        
       
#    clf_RF = RandomForestClassifier(n_estimators=1000, max_depth=3, max_features=1, max_leaf_nodes=2, class_weight='balanced_subsample')
#    clf_RF.fit(X_train_fs, y_train)
#    y_proba_RF = clf_RF.predict_proba(X_test_fs)
#    f1_RF = f1_score(y_test, y_proba_RF[:, 1] > 0.5)
#    print('f1 (random forest): {}'.format(f1_RF))
    
    
    
    
#    
#    
#    
#    y_pred_random = np.random.randint(2, size=len(y_test))
#    
#    f1_LR = f1_score(y_test, y_proba_LR[:, 1] > 0.5)
#    f1_RF = f1_score(y_test, y_proba_RF[:, 1] > 0.5)
#    f1_Random = f1_score(y_test, y_pred_random)
#    print('f1 (logistic regression): {}'.format(f1_LR))
#    print('f1 (random forest): {}'.format(f1_RF))
#    print('f1 (random): {}'.format(f1_Random))


# X_train = X[:-20]
# X_test = X[-20:]
# transformation_pipe = transformations.transform_input(X_train)
#
# X_train = transformation_pipe.fit_transform(X_train)
# X_test_transformed = transformation_pipe.transform(X_test)

# imputation_pipe.fit(X_rest)
# X_imputed = imputation_pipe.transform(X_null)


# mask = (X.attr.isnull()) ^ (X.age_o.isnull()) ^ (X.go_date.isnull())
# X_null = X[mask]
# X_rest = X[~mask]

# X = imputation_pipe.fit_transform(X)
#
# imputation = make_column_transformer(
#     (SimpleImputer(strategy='mean'),
#      ['pf_o_att', 'pf_o_fun', 'pf_o_amb',
#       'pf_o_sin', 'pf_o_sha', 'pf_o_int',
#       'attr', 'sinc', 'fun', 'int', 'amb',
#       'int_corr']), remainder='passthrough')
# X_imputed = pd.DataFrame(imputation.fit_transform(X))




#    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#             "Naive Bayes", "QDA"]
#
#    classifiers = [
#        KNeighborsClassifier(3),
#        SVC(kernel="linear", C=0.025, probability=True),
#        SVC(gamma=2, C=1, probability=True),
#        GaussianProcessClassifier(1.0 * RBF(1.0)),
#        DecisionTreeClassifier(max_depth=5),
#        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#        MLPClassifier(alpha=1),
#        AdaBoostClassifier(),
#        GaussianNB(),
#        QuadraticDiscriminantAnalysis()]
#
#    for name, clf in zip(names, classifiers):
#        clf.fit(X_train, y_train)
#        y_proba = clf.predict_proba(X_test)
#        f1 = f1_score(y_test, y_proba[:, 1] > 0.5)
#        print(f'f1 ({name}): {f1}')
    
    