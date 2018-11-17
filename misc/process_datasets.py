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
pathinput = input('Enter datasets repository path:')
if not pathinput:
    pathinput = '/media/romulo/C4B4FA64B4FA57FE//datasets//'
type_ext = input('Enter extension type (default=.csv):')
if not type_ext:
    type_ext = '.csv'
pathoutput = input('Enter repository to save:')
if not pathoutput:
    pathoutput = '/media/romulo/C4B4FA64B4FA57FE//datasets_prep//'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)

# Loop and prepare dataset and save
# in output repo
for file_path in tqdm(files_list, unit='files'):
    file_name = file_path.split('/')[-1]

    #print(file_name)
    # past 0.05 and 0.1

    dataset_out = process_file(file_path, 0.14, 0.18)

    dataset_out.to_csv(pathoutput + file_name)
