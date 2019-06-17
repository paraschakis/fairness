#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectFromModel


import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='monospace', style='normal', size=10)

df = pd.read_csv('dataset_experimental_working.csv', sep=';')

X = df.iloc[:, 3:-2]
X = pd.get_dummies(X)
X = X.drop(['gender_male', 'gender_female', 'dec'], 1)

y = df.iloc[:, -3].values

# FILL MISSING VALUES
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X.iloc[:, :])
X.iloc[:, :] = imp.transform(X.iloc[:, :])

# RE-SCALE TO NORMALIZE
scl = StandardScaler()
scl = scl.fit(X.iloc[:, :])
X.iloc[:, :] = scl.transform(X.iloc[:, :])

clf_LR = LogisticRegression()
clf_RF = RandomForestClassifier()
y_real = []
y_proba_LR = []
y_proba_RF = []
y_proba_RANDOM = []


def find_best_params(classifier, X_train, y_train, name):
    if name == 'LR':
        params = { 'penalty': ['l1','l2'], 'C': [0.1,1,10,100,1000] }
    elif name == 'RF':
        params = { 
            'n_estimators': [1, 10, 50, 200, 500],
            'max_features': ['auto', 'sqrt', 'log2'] 
        }
    clf = GridSearchCV(classifier, params, cv=5, scoring='f1')
    clf.fit(X_train, y_train)
    return clf.best_params_
    

# TRAIN
for wave in range(1, 22):
    X_train = X[ df.wave != wave ]
    y_train = y[ df.wave != wave ]
    X_true  = X[ df.wave == wave ]
    y_true  = y[ df.wave == wave ]
    
    # VALIDATION SET
    X_train, _X_val, y_train, _y_val = train_test_split(X_train, y_train, test_size=.2)
    X_val_train, X_val_true, y_val_train, y_val_true = train_test_split(_X_val, _y_val, test_size=.2)
    
    # LOGISTIC REGRESSION
    _best_params = find_best_params(clf_LR, X_val_train, y_val_train, 'LR')
    clf_LR = LogisticRegression(**_best_params)
    clf_LR = clf_LR.fit(X_train, y_train)
    model = SelectFromModel(clf_LR, prefit=True, threshold=0.1)
    X_train_new = X_train.loc[:, X_train.columns[model.get_support(indices=True)]]
    X_true_new = model.transform(X_true)
    clf_LR = clf_LR.fit(X_train_new, y_train)
    y_pred_LR = clf_LR.predict_proba(X_true_new)
    y_proba_LR.append(y_pred_LR[:,1])
    
    # RANDOM FOREST    
    _best_params = find_best_params(clf_RF, X_val_train, y_val_train, 'RF')
    clf_RF = RandomForestClassifier(**_best_params)
    y_pred_RF = clf_RF.fit(X_train, y_train).predict_proba(X_true)
    y_proba_RF.append(y_pred_RF[:,1])
    
    y_proba_RANDOM.append(np.random.randint(2, size=len(y_true)))
    
    y_real.append(y_true)
    

y_real   = np.concatenate(y_real)
y_rec_LR = np.concatenate(y_proba_LR)
y_rec_RF = np.concatenate(y_proba_RF)
y_rec_RANDOM = np.concatenate(y_proba_RANDOM)


# PLOT LOGISTIC REGRESSION
precision, recall, thresholds = precision_recall_curve(y_real, y_rec_LR)
label = 'AUC=%.4f Logistic Regression' % (auc(recall, precision))
plt.plot(recall, precision, label=label, linewidth=1)

# PLOT RANDOM FOREST
precision, recall, _ = precision_recall_curve(y_real, y_rec_RF)
label = 'AUC=%.4f Random Forest' % (auc(recall, precision))
plt.plot(recall, precision, label=label, linewidth=1)

# PLOT BASELINE
p = np.sum(y_real)
n = len(y_real) - p
baseline = p / (p + n)
label = 'AUC=%.4f Baseline' % baseline
plt.setp(plt.plot([0,1], [baseline,baseline], label=label, linewidth=1), ls='--', color='grey', alpha=0.5)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR AUC')
plt.legend(loc="lower left", prop=font)
plt.show()

# F1 SCORE
threshold  = p / (p + n)
f1_LR = f1_score(y_real, y_rec_LR >= threshold)
f1_RF = f1_score(y_real, y_rec_RF >= threshold)
f1_RANDOM = f1_score(y_real, y_rec_RANDOM)
print("F1 score LR: %4f" % f1_LR)
print("F1 score RF: %4f" % f1_RF)
print("F1 score RANDOM: %4f" % f1_RANDOM)


