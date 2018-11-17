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
	dataset_name = file_name.split('.')[0]
	logging.info('Dataset: {:}'.format(dataset_name))	
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
	mf = MetaFeatures(dataset_name=re.sub('.csv', '', file_name), metric='rmse')
	mf.fit(X, y)
	meta_instance = mf.get_params()
	#
	meta_list.append(meta_instance)

meta = pathoutput + 'meta_features.csv'

logging.info('Writing file in {:}'.format(meta))
pd.DataFrame(meta_list).dropna().set_index('dataset').to_csv(meta)
logging.info('Done.')

