# coding=utf-8

import sys
sys.path.append("..")

import re
import pandas as pd
import numpy as np

from glob import glob
from automl.preprocessing import process_file
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')

# Inputs
pathinput = '/media/romulo/C4B4FA64B4FA57FE//datasets//'
type_ext = '.csv'
pathoutput = '/media/romulo/C4B4FA64B4FA57FE//datasets_prep//'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)

def clean_data(file_path):

    def remove_brackets(file_path):
        with open(file_path) as f:
            data = f.read()
        with open(file_path, 'w') as f:
            f.write(re.sub('{|}', '', data))

    def remove_interrogation(file_path):
        try:
            return pd.read_csv(file_path).replace(
        {'?': np.nan}).apply(pd.to_numeric, errors='ignore')
        except Exception as e:
            return pd.read_csv(file_path)

    def remove_empty_spaces(x):
        try:
            return float(x.strip().split(' ')[-1])
        except:
            return x

    # File preprocess
    remove_brackets(file_path)
    dataset = remove_interrogation(file_path)
    dataset = dataset.applymap(remove_empty_spaces)

    return dataset

unique_values_list = list()
for file_path in tqdm(files_list):
	dataset = clean_data(file_path)
	n = dataset.shape[0]
	for col in dataset.columns:
		if dataset[col].dtype.kind not in 'bifc':
			n_unique = len(dataset[col].unique())
			unique_values_list.append(n_unique / n)

np.savetxt('dist_cat.txt', np.array(unique_values_list))

nan_values_list = list()
for file_path in tqdm(files_list):
	dataset = clean_data(file_path)
	n = dataset.shape[0]
	for col in dataset.columns:
		n_nan = dataset[col].isna().sum()
		if n_nan:
			nan_values_list.append(n_nan / n)

np.savetxt('dist_nan.txt', np.array(nan_values_list))