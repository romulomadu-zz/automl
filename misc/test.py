
import sys
import os
import multiprocessing
import pandas as pd
sys.path.append("..")

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate
from config import grid_params, random_params, bayes_params, nmse

path = '../datasets_preprocessed/disclosure_x_noise.csv'
dataset = pd.read_csv(path, index_col=0)

def make_search(X, y, params, method='grid', random_state=0):
    num_cores = multiprocessing.cpu_count() - 1
    n_features = X.shape[1]
    model = SVR()
    if method=='grid':
        search = GridSearchCV(model, **params, n_jobs=num_cores)
    elif method=='random':
        search = RandomizedSearchCV(model, **params, n_jobs=num_cores) 
    elif method=='bayes':
        search = BayesSearchCV(model, **params, n_jobs=num_cores)
    search.fit(X, y)

    return search

# Select search type
search_type = sys.argv[1]
if search_type == 'grid':
	search_params = grid_params
elif search_type == 'bayes':
	search_params = bayes_params
elif search_type == 'random':
	search_params = random_params
else:
	search_type = 'grid'
	search_params = grid_params

# Separe features from target
X_train = dataset.iloc[:,:-1].values
y_test =  dataset.iloc[:,-1].values

meta_instance = {'dataset': 'disclosure_x_noise'}

# Train SVR models with:
# - Grid Search
# - Random Search
# - Bayesian Search
model = make_search(X_train, y_train, search_params(), method=search_type)
meta_instance['p_{:}_search'.format(search_type)] = model.best_params_
meta_instance['nmse_{:}_search'.format(search_type)] = abs(model.best_score_)
meta_list.append(meta_instance)
print(meta_instance)

param_cross = {
	'estimator': SVR(**model.best_params_),
	'X': X_train,
	'y': y_train,
	'scoring': make_scorer(nmse, greater_is_better=False),
	'verbose': 1,
	'cv': 10
}

cross = cross_validate(**param_cross)
print(cross.best_params_, cross.best_score_)