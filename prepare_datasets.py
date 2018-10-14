import re
import pandas as pd
import numpy as np

from glob import glob
from automl.file_preparation import PrepareDataset

# Inputs
pathinput = input('Enter datasets repository path:')
if not pathinput:
	pathinput = '/media/romulo/C4B4FA64B4FA57FE//bases//'
type_ext = input('Enter extension type (default=.csv):')
if not type_ext:
	type_ext = '.csv'
pathoutput = input('Enter repository to save:')
if not pathoutput:
	pathoutput = '/media/romulo/C4B4FA64B4FA57FE//bases_preparadas//'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)

# Loop and prepare dataset and save
# in output repo
for file_path in files_list:
	file_name = file_path.split('/')[-1]
	print(file_name)

	# For files downloaded form OpenML
	with open(file_path) as f:
	    data = f.read()
	with open(file_path, 'w') as f:
	    f.write(re.sub('{|}','', data))

	try:
	    dataset = pd.read_csv(file_path).replace({'?':np.nan}).apply(pd.to_numeric, errors='ignore')
	except:
	    dataset = pd.read_csv(file_path)

	def clean_str(x):
	    try:
	        return float(x.strip().split(' ')[-1])
	    except:
	        return x

	dataset = dataset.applymap(clean_str)

	prepdata = PrepareDataset(n_unique_values_treshold=.01, 
                 na_values_ratio_theshold=.10)
	X = dataset.iloc[:,:-1]
	y =  dataset.iloc[:,-1]
	dataset_out = prepdata.fit_transform(X, y)

	dataset_out.to_csv(pathoutput+file_name)







