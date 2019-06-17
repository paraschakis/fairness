from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


#clf_params = [
#    'LogisticRegression': {
#                {'C': [0.1,1,10,100,1000]}}
#    ]

#LogisticRegression = [
#                {'C': [0.1,1,10,100,1000]}]

#def add_prefix(params, prefix):
#    for index, grid in enumerate(params):
#        params[index] = {prefix+'__'+str(k): v for k, v in grid.items()}
#    return params

#def optimize_threshold(y_proba, y_train):
#    best_threshold = 0.5
#    best_f1 = f1_score(y_test, y_proba[:, 1] > 0.5)
#    for threshold in np.linspace(0.1, 0.5, num=50):
#        f1 = f1_score(y_test, y_proba[:, 1] > threshold)
#        if f1 > best_f1:
#            best_threshold = threshold
#            best_f1 = f1
#    return best_threshold, best_f1

#def rebalance_params():
#    random = RandomUnderSampler(random_state=0)
#    enn = EditedNearestNeighbours(random_state=0, n_jobs=-1)
#    params = [{'rebalance': [random, enn, None]}]
#    return params
#
#def reduce_dim_params(n_features):
#    params = [
#                {'reduce_dim': [SelectKBest()],
#                 'reduce_dim__k': [10, 50, 100, n_features]},
##                {'reduce_dim': [PCA(iterated_power=7)],
##                 'reduce_dim__n_components': [2, 4, 8]},
##                {'reduce_dim': [None]}
#    ]
#    return params

def logisticregression():
    params = [{'C': [0.01, 0.05, 0.1, 0.5, 1.0],
               'class_weight': ['balanced', None],
               'penalty': ['l1', 'l2']}]
    return params
#    return add_prefix(params, 'classify')

def randomforestclassifier():
    params = [
            {'n_estimators': [100, 500, 1500],
             'max_depth': [2, 3, 6],
#             'max_features': ['auto', 'log2'],
             'max_leaf_nodes': [3, 10],
#             'min_samples_leaf': [3, 5],
#             'min_samples_split': [3, 5],
             'criterion': ['entropy', 'gini'],
#             'class_weight': ['balanced', 'balanced_subsample', None]
            }
        ]
    return params

def kneighborsclassifier():
    params = [{'n_neighbors': [1, 3, 5, 10, 30],
               'weights': ['distance', 'uniform']
            }
        ]
    return params   

def mlpclassifier():
    params = [
            {'learning_rate': ['constant', 'invscaling', 'adaptive'],
             'alpha': [0.0001, 0.001, 0.01],
             'early_stopping': [True, False]
            }
        ]
    return params

def xgbclassifier():
    params = [
            {
            'scale_pos_weight': [1, 5, 10],
             'n_estimators': [100, 300, 1000],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [1, 3, 7],
             'gamma': [0, 1, 5]
            }
        ]
    return params

def gradientboostingclassifier():
    params = [
            {
             'n_estimators': [300, 1000, 1500],
             'learning_rate': [0.01, 0.1],
             'max_depth': [1, 3, 6],
             'loss': ['deviance','exponential']
            }
        ]
    return params

def linearsvc():
    params = [
            {
             'C': [1, 5, 10]
            }
        ]
    return params


def rebalance_grid(pipeline):
    rand_undersampler = RandomUnderSampler(random_state=0)
    rand_oversampler = RandomOverSampler(random_state=0)
    smote = SMOTE(n_jobs=-1)
    enn = EditedNearestNeighbours(random_state=0, n_jobs=-1)
    params = [{'rebalance': [rand_undersampler, rand_oversampler, enn, None]}]
#    params = [{'rebalance': [None]}]f1
    return GridSearchCV(pipeline, param_grid=params,
                        n_jobs=-1, cv=4, scoring='roc_auc', verbose=0)
    
    
def reduce_dim_grid(pipeline):
    params = [
                {'reduce_dim': [SelectKBest()],
                 'reduce_dim__k': [5, 15, 30, 'all']},
#                {'reduce_dim': [PCA(iterated_power=7)],
#                 'reduce_dim__n_components': [2, 4, 8]},
    ]
    return GridSearchCV(pipeline, param_grid=params,
                        n_jobs=-1, cv=4, scoring='roc_auc', verbose=0)

def classifier_grid(clf):
    clf_name = clf.__class__.__name__.lower()
    try:
        params = globals()[clf_name]()  # populate params from function
    except KeyError: # if no param_grid provided, don't do grid search
        return clf
    else:
        return GridSearchCV(clf, param_grid=params,
                        n_jobs=-1, cv=3, scoring='roc_auc', verbose=1)
    

#def define_total_grid(pipeline, n_features):
#    clf_name = pipeline.named_steps['classify'].__class__.__name__.lower()
#    clf_params = globals()[clf_name]  # map the name to the right function
#    param_grid = [
##                  *rebalance_params(), 
#                  *reduce_dim_params(n_features), 
#                  *clf_params()]
#    params_combined = {k: v for d in param_grid for k, v in d.items()}
#    grid = GridSearchCV(pipeline, param_grid=params_combined,
#                        n_jobs=-1, cv=3, scoring='f1', verbose=1)
    
#    random_search = RandomizedSearchCV(pipeline, param_distributions=params_combined,
#                                   n_iter=20, cv=3)
    
#    return grid


#clf = [
#        {'C': [0.1,1,10,100,1000]},
#        {'bla': ['a', 'b']}
#        ]

#print (add_prefix(clf, 'classify'))
