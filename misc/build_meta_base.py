# coding=utf-8

import sys
import os
import re
sys.path.append("..")

import pandas as pd
import numpy as np
import multiprocessing

from glob import glob
from automl.meta_features import MetaFeatures
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from skopt import BayesSearchCV
from config import grid_params, random_params, bayes_params, nmse
from automl.preprocessing import process_file
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
cpath = os.getcwd()
logging.basicConfig(
	filename=re.sub('misc', '', cpath) + '/logs/log.log',
	format='%(asctime)s %(levelname)-8s %(message)s', 
	level=logging.INFO, 
	datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Application Run: build_meta_base')


def make_search(X, y, params, method='grid', random_state=0):
    num_cores = multiprocessing.cpu_count() - 1
    n_features = X.shape[1]
    model = SVR()
    if method=='grid':
        search = GridSearchCV(model, **params, n_jobs=num_cores, return_train_score=False)
    elif method=='random':
        search = RandomizedSearchCV(model, **params, n_jobs=num_cores, return_train_score=False) 
    elif method=='bayes':
        search = BayesSearchCV(model, **params, n_jobs=num_cores, return_train_score=False)
    search.fit(X, y)

    return search
	
# With inputs
#prepinput = input('Datasets are preprocessed? (yes or no): ')
#if not prepinput:
#	prepinput = 'yes'
#pathinput = input('Enter datasets repository path:')
#if not pathinput:
#	pathinput = re.sub('misc', '', cpath) + '/datasets_preprocessed/'
#type_ext = input('Enter extension type (default=.csv):')
#if not type_ext:
#	type_ext = '.csv'
#pathoutput = input('Enter repository to save "metabase.csv":')
#if not pathoutput:
#	pathoutput = re.sub('misc', '', cpath) + '/meta_db/'

np.random.seed(0)

# Without inputs
prepinput = 'yes'
type_ext = '.csv'
pathinput = re.sub('misc', '', cpath) + '/datasets_preprocessed/'
pathoutput = re.sub('misc', '', cpath) + '/meta_db/'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)
meta_list = list()

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
	
# Loop and prepare dataset and save
# in output repo
for file_path in tqdm(files_list, unit='files'):
	file_name = file_path.split('/')[-1]
	dataset_name = re.sub('.csv', '', file_name)

	logging.info('Dataset: {:}'.format(dataset_name))	
	is_prep = prepinput == 'yes'
	if is_prep:
		dataset = pd.read_csv(file_path, index_col=0)
	else:
		dataset = process_file(file_path)

	# Separe features from target
	X_train = dataset.iloc[:,:-1].values
	y_train =  dataset.iloc[:,-1].values

	meta_instance = {'dataset': dataset_name}

	# Train SVR models with:
	# - Grid Search
	# - Random Search
	# - Bayesian Search
	logging.info('Search method: {:}.'.format(search_type))
	model = make_search(X_train, y_train, search_params(), method=search_type)
	meta_instance['p_{:}_search'.format(search_type)] = model.best_params_
	meta_instance['mse_{:}_search'.format(search_type)] = abs(model.best_score_)
	logging.info('MSE: {:}.'.format(abs(model.best_score_)))
	y_pred = model.predict(X_train)
	meta_instance['nmse_{:}_search'.format(search_type)] = nmse(y_train, y_pred)
	logging.info('NMSE: {:}.'.format(nmse(y_train, y_pred)))
	meta_list.append(meta_instance)

# Save results
meta = pathoutput + 'meta_{:}.csv'.format(search_type)
logging.info('Writing file in {:}'.format(meta))
pd.DataFrame(meta_list).dropna().set_index('dataset').to_csv(meta)
logging.info('Done.')