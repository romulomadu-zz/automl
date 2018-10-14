import pandas as pd
import numpy as np

from glob import glob
from automl.meta_features import MetaFeatures

# Inputs
pathinput = input('Enter datasets repository path:')
if not pathinput:
	pathinput = '/media/romulo/C4B4FA64B4FA57FE//bases_preparadas//'
type_ext = input('Enter extension type (default=.csv):')
if not type_ext:
	type_ext = '.csv'
pathoutput = input('Enter repository to save "meta_features.csv":')
if not pathoutput:
	pathoutput = '/media/romulo/C4B4FA64B4FA57FE//meta_features//'

# Get file in directory
files_path = pathinput + '*' + type_ext
files_list = glob(files_path)

meta_list = list()
# Loop and prepare dataset and save
# in output repo
for file_path in files_list:
	file_name = file_path.split('/')[-1]
	print(file_name)

	dataset = pd.read_csv(file_path)
	X = dataset.iloc[:,:-1].values
	y =  dataset.iloc[:,-1].values
	mf = MetaFeatures(dataset_name=file_name.split('.')[0], metric='rmse')
	mf.fit(X, y)
	meta_list.append(mf.get_params())

pd.DataFrame(meta_list).to_csv(pathoutput+'meta_features.csv')