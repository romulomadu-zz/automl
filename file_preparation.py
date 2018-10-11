import re
import pandas as pd
import numpy as np
import random

from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

__author__ = 'Rômulo Rodrigues <romulomadu@gmail.com>'
__version__ = '0.1'


class PrepareDataset(object):

    def __init__(self, n_unique_values_treshold=.15, 
                 na_values_ratio_theshold=.15):
        self.n_unique_values_treshold = n_unique_values_treshold
        self.na_values_ratio_theshold = na_values_ratio_theshold
        self.dataset = pd.DataFrame()

    def fit(self, dataset):
        self.dataset = dataset
        self._remove_na()
        self._remove_categorical_up_thresh()
        self._imputer()
        self._get_dummies()
    
    def fit_transform(self, dataset):
        self.fit(dataset)
        
        return self.dataset
    
    def _remove_na(self):        
        na_cond = self.dataset.apply(na_rate, axis=0) < self.na_values_ratio_theshold
        true_cols = self.dataset.columns[na_cond.values]        
        self.dataset = self.dataset[true_cols]
    
    def _remove_categorical_up_thresh(self):        
        for col in self.dataset.columns:
            is_numeric = self.dataset[col].dtype.kind in 'bifc'
            unique_tresh = unique_threshold(self.dataset[col],
                                           self.n_unique_values_treshold)            
            if is_numeric:
                if unique_tresh:
                    self.dataset[col] = self.dataset[col].astype('category')
            else:
                if unique_tresh:
                    self.dataset[col] = self.dataset[col].astype('category')
                else:
                    self.dataset.drop(col, inplace=True, axis=1)
    
    def _imputer(self):
        for col in self.dataset.columns:
            is_numeric = self.dataset[col].dtype.kind in 'bifc'
            has_na = na_rate(self.dataset[col]) > .0
            
            if is_numeric and has_na:
                median = self.dataset[col].median()
                self.dataset[col].fillna(median, inplace=True)            
            else:
                if has_na:
                    mod = self.dataset[col].mode()[0]
                    self.dataset[col].fillna(mod, inplace=True)
    
    def _get_dummies(self):
        self.dataset = pd.get_dummies(self.dataset)
    

def na_rate(series):
    return series.isna().sum()/series.shape[0]


def unique_threshold(series, threshold=.5):
    return len(series.unique())/len(series) < threshold


def main():
    alphabet = 'abcdefghijklmnopqrstuyxwz'
    abc = [1,2,3,4,5]
    cat = [abc[i%5] for i in range(100)]
    num = np.random.rand(98).tolist() + [np.nan, np.nan]

    df = pd.DataFrame(list(zip(cat,num,num)))

    print(f'Before preparation:\n{df.tail()}\n')

    prepdata = PrepareDataset()

    df = prepdata.fit_transform(df)
    
    print(f'After preparation:\n{df.tail()}')

if __name__ == '__main__':
    main()
