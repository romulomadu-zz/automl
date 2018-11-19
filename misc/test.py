
import sys
import os
import multiprocessing
import pandas as pd
sys.path.append("..")

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from config import grid_params, nmse

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


# Separe features from target
X = dataset.iloc[:,:-1].values
y =  dataset.iloc[:,-1].values

meta_instance = {'dataset': 'disclosure_x_noise'}

# Train SVR models with:
# - Grid Search
# - Random Search
# - Bayesian Search
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = make_search(X_train, y_train, grid_params(), method='grid')
y_pred = model.predict(X_test)
meta_instance['p_grid_search'] = model.best_params_
meta_instance['nmse_grid_search'] = nmse(y_pred, y_test)

print(meta_instance)