# coding=utf-8

import sys
import os
import re
import logging
sys.path.append("..")

import pandas as pd
import numpy as np
import multiprocessing

from glob import glob
from automl.meta_features import MetaFeatures
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from skopt import BayesSearchCV
from config import grid_params, random_params, bayes_params, nmse
from automl.preprocessing import process_file
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

cpath = os.getcwd()

logging.basicConfig(filename=re.sub('scripts', '', cpath) + '/logs/log.log',level=logging.INFO)
logging.info('Application Run: build_meta_base')


def make_search(X, y, params, method='grid', random_state=0):
    num_cores = multiprocessing.cpu_count() - 1
    n_features = X.shape[1]
    model = SVR()
    if method=='grid':
        search = GridSearchCV(model, **params, n_jobs=num_cores)
    elif method=='random':
        search = 	RandomizedSearchCV(model, **params, n_jobs=num_cores)        
    elif method=='bayes':
        search = BayesSearchCV(model, **params, n_jobs=num_cores)
    search.fit(X, y)

    return search


# Inputs
prepinput = input('Datasets are preprocessed? (yes or no): ')
if not prepinput:
	prepinput = 'yes'
pathinput = input('Enter datasets repository path:')
if not pathinput:
	#pathinput = '/media/romulo/C4B4FA64B4FA57FE//datasets_prep//'
	pathinput = re.sub('scripts', '', cpath) + '/datasets_preprocessed/'
type_ext = input('Enter extension type (default=.csv):')
if not type_ext:
	type_ext = '.csv'
pathoutput = input('Enter repository to save "meta.csv":')
if not pathoutput:
#	pathoutput = '/media/romulo/C4B4FA64B4FA57FE//meta_db//'
	pathoutput = re.sub('scripts', '', cpath) + '/meta_db/'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)
meta_list = list()

# Loop and prepare dataset and save
# in output repo
for file_path in tqdm(files_list, unit='files'):
	file_name = file_path.split('/')[-1]
	logging.info(f'Dataset: {file_name.split('.')[0]}')	
	is_prep = prepinput == 'yes'
	if is_prep:
		dataset = pd.read_csv(file_path, index_col=0)
	else:
		dataset = process_file(file_path)

	# Separe features from target
	X = dataset.iloc[:,:-1].values
	y =  dataset.iloc[:,-1].values

	# Get Meta features
	logging.info('Getting meta-features.')
	mf = MetaFeatures(dataset_name=file_name.split('.')[0], metric='rmse')
	mf.fit(X, y)
	meta_instance = mf.get_params()

	# Train SVR models with:
	# - Grid Search
	# - Random Search
	# - Bayesian Search
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	#logging.info('Grid search.')
	#model = make_search(X_train, y_train, grid_params(), method='grid')
	#y_pred = model.predict(X_test)
	#meta_instance['p_grid_search'] = model.best_params_
	#meta_instance['nmse_grid_search'] = nmse(y_pred, y_test)

	#logging.info('Random Search.')
	#model = make_search(X_train, y_train, random_params(), method='random')
	#y_pred = model.predict(X_test)
	#meta_instance['p_random_search'] = model.best_params_
	#meta_instance['nmse_random_search'] = nmse(y_pred, y_test)

	logging.info('Bayes Search.')
	model = make_search(X_train, y_train, bayes_params(), method='bayes')
	y_pred = model.predict(X_test)
	meta_instance['p_bayes_search'] = model.best_params_
	meta_instance['nmse_bayes_search'] = nmse(y_pred, y_test)

	meta_list.append(meta_instance)

meta = pathoutput + 'meta.csv'

logging.info(f'Writing file in {meta}')
pd.DataFrame(meta_list).dropna().set_index('dataset').to_csv(meta)
logging.info('Done.')

