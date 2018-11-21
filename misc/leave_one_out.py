import numpy as np 
import pandas as pd 
import sys
import os
import re
import ast

sys.path.append("..")

from automl.meta_learning import KNearestNeighbors
from sklearn.metrics import cross_validate
from sklearn.svm import SVR
from config import nmse

cpath = os.getcwd()
path = re.sub('misc', '', cpath) + 'meta_db/'

metafeatures = pd.read_csv(path + 'meta_features.csv', index_col=0)
metatargets = pd.read_csv(path + 'meta_bayes.csv', index_col=0)
metabase = pd.merge(metafeatures, metatargets, left_index=True, right_index=True)

X = metabase.drop(['nmse_bayes_search', 'p_bayes_search'], axis=1)
y = metabase.p_bayes_search

idx = 30

dataset_name = y.index[idx]
real_param = ast.literal_eval(y.iloc[idx])

x = X.iloc[idx].values
X = X.drop(X.index[idx]).values
y = y.drop(y.index[idx]).values

knn = KNearestNeighbors(k=5)
knn.fit(X, y)
candidates = knn.predict(x)

dataset = pd.read_csv(re.sub('misc', '', cpath) + 'datasets_preprocessed/' + dataset_name + '.csv')

X_train = dataset.iloc[:,:-1]
y_train = dataset.iloc[:,-1]

learner = SVR(**real_param)
model = cross_validate(learner, X_train, y_train, cv=10, scoring=make_scorer(nmse, greater_is_better=False))

print(dataset_name+'\n')
print('Real : {:}'.format(abs(model.best_score_)))

i = 0
for c in candidates:
	i += 1
	params = ast.literal_eval(c)
	learner = SVR(**real_param)
	model = cross_validate(learner, X_train, y_train, cv=10, scoring=make_scorer(nmse, greater_is_better=False))
	print('candidate {:}: {:}'.format(i, abs(model.best_score_)))






