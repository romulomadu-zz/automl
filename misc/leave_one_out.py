import numpy as np 
import pandas as pd 
import sys
import os
import re
import ast

sys.path.append("..")

from automl.meta_learning import KNearestNeighbors
from sklearn.model_selection import train_test_split
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

X_ = dataset.iloc[:,:-1]
y_ = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=0)

clf = SVR(**real_param)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(dataset_name+'\n')

print('Real : {:}'.format(nmse(y_pred, y_test.values)))

i = 0
for c in candidates:
	i += 1
	params = ast.literal_eval(c)
	clf = SVR(**params)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print('candidate 1: {:}'.format(nmse(y_pred, y_test.values)))






