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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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
        search = GridSearchCV(model, **params, n_jobs=num_cores)
    elif method=='random':
        search = RandomizedSearchCV(model, **params, n_jobs=num_cores) 
    elif method=='bayes':
        search = BayesSearchCV(model, **params, n_jobs=num_cores)
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

# Without inputs
prepinput = 'yes'
type_ext = '.csv'
pathinput = re.sub('misc', '', cpath) + '/datasets_preprocessed/'
pathoutput = re.sub('misc', '', cpath) + '/meta_db/'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)
meta_list = list()

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
	X = dataset.iloc[:,:-1].values
	y =  dataset.iloc[:,-1].values

	meta_instance = {'dataset': dataset_name}

	# Train SVR models with:
	# - Grid Search
	# - Random Search
	# - Bayesian Search
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	logging.info('Random Search.')
	model = make_search(X_train, y_train, random_params(), method='random')
	y_pred = model.predict(X_test)
	meta_instance['p_random_search'] = model.best_params_
	meta_instance['nmse_random_search'] = nmse(y_pred, y_test)

	meta_list.append(meta_instance)

meta = pathoutput + 'meta_random.csv'

logging.info('Writing file in {:}'.format(meta))
pd.DataFrame(meta_list).dropna().set_index('dataset').to_csv(meta)
logging.info('Done.')