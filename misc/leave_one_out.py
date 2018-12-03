import numpy as np 
import pandas as pd 
import sys
import os
import re
import ast

sys.path.append("..")

from automl.meta_learning import KNearestNeighbors
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
from config import nmse
from tqdm import tqdm

cpath = os.getcwd()
path = re.sub('misc', '', cpath) + 'meta_db/'

meta_features = pd.read_csv(path + 'meta_features.csv', index_col=0)
meta_targets = pd.read_csv(path + 'meta_grid.csv', index_col=0)
meta_base = pd.merge(meta_features, meta_targets, left_index=True, right_index=True)

#Set
feature_set = [
	'c1',
	'c2',
	'c3',
	'c4',
	'l1_a',
	'l2_a',
	'l3_a',
	's1',
	's2',
	's3',
	's4',
	't2'
]

features = meta_base.loc[:, feature_set]
params = meta_base.p_grid_search

loo_list = list()

for idx in tqdm(range(features.shape[0])):
	instance_result = dict()

	dataset_name = params.index[idx]
	real_param = ast.literal_eval(params.iloc[idx])
	# Store dataset name
	instance_result['Dataset'] = dataset_name 

	meta_example = features.iloc[idx].values + .0 # convert bool to float
	features_filt = features.drop(features.index[idx]).values
	params_filt = params.drop(params.index[idx]).values
	instance_result['Grid'] = meta_base.mse_grid_search[idx]

	knn = KNearestNeighbors(k=1)
	knn.fit(features_filt, params_filt)
	candidates = knn.predict(meta_example)

	dataset = pd.read_csv(re.sub('misc', '', cpath) + 'datasets_preprocessed/' + dataset_name + '.csv')

	X_train = dataset.iloc[:,:-1]
	y_train = dataset.iloc[:,-1]

	#learner = SVR(**real_param)
	#model = cross_validate(learner, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

	i = 0
	for c in candidates:
		i += 1
		param = ast.literal_eval(c)
		learner = SVR(**param)
		model = cross_validate(learner, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=3, return_train_score=False)
		instance_result['Proposal'] =+ abs(model['test_score']).mean()

	instance_result['Proposal'] = instance_result['Proposal'] / i

	loo_list.append(instance_result)

pd.DataFrame(loo_list).to_csv('loo_table.csv')





