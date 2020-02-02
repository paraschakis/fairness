from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest


def logisticregression():
    params = [{'C': [0.01, 0.05, 0.1, 0.5, 1.0],
               'class_weight': ['balanced', None],
               'penalty': ['l1', 'l2']}]
    return params


def randomforestclassifier():
    params = [
            {'n_estimators': [100, 500, 1500],
             'max_depth': [2, 3, 6],
             'max_leaf_nodes': [3, 10],
             'criterion': ['entropy', 'gini'],
            }
        ]
    return params


def xgbclassifier():
    params = [
            {
            'scale_pos_weight': [1, 5, 10],
             'n_estimators': [100, 300, 1000],
             'gamma': [0, 1, 5]
            }
        ]
    return params


def rebalance_grid(pipeline):
    rand_undersampler = RandomUnderSampler(random_state=0)
    rand_oversampler = RandomOverSampler(random_state=0)
    smote = SMOTE(n_jobs=-1)
    enn = EditedNearestNeighbours(random_state=0, n_jobs=-1)
    params = [{'rebalance': [rand_undersampler, rand_oversampler, enn, None]}]
    return GridSearchCV(pipeline, param_grid=params,
                        n_jobs=-1, cv=4, scoring='roc_auc', verbose=0)


def reduce_dim_grid(pipeline):
    params = [
                {'reduce_dim': [SelectKBest()],
                 'reduce_dim__k': [5, 15, 30, 'all']},
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
