# coding=utf-8

import re
import pandas as pd
import numpy as np
import random

from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


__author__ = 'Rômulo Rodrigues <romulomadu@gmail.com>'
__version__ = '0.1.0'


class PrepareDataset(object):

    def __init__(self, n_unique_values_treshold=.02, 
                 na_values_ratio_theshold=.10):
        self.n_unique_values_treshold = n_unique_values_treshold
        self.na_values_ratio_theshold = na_values_ratio_theshold
        self.dataset = pd.DataFrame()

    def fit(self, X):
        self.dataset = X
        self.__prepare_dataset()
    
    def fit_transform(self, X, y):
        self.fit(X) 
        y = imputer(y)
        y = y.values.reshape((y.shape[0],1))
        y_scaled = MinMaxScaler().fit_transform(y)
        y_scaled = pd.Series(y_scaled.T[0], name='Target')     

        return pd.concat([self.dataset, y_scaled], axis=1)    
    
    def __get_dummies(self):
        try:
            self.dataset = pd.get_dummies(self.dataset)
        except:
            pass    

    def __prepare_dataset(self):
        for col in self.dataset.columns:
            series = self.dataset[col].copy()

            if is_sequential_like_index(series):
                self.dataset.drop(col, inplace=True, axis=1)
                continue

            if is_na_ratio_up_thresh(series, self.na_values_ratio_theshold):
                self.dataset.drop(col, inplace=True, axis=1)
                continue

            if is_categorical(series, self.n_unique_values_treshold):
                if is_categorical_up_thresh(series, self.n_unique_values_treshold):
                    self.dataset.drop(col, inplace=True, axis=1)
                    continue
                else:
                    self.dataset.at[:, col] = self.dataset[col].astype('category').values

            self.dataset.at[:, col] = imputer(series).values

        self.__get_dummies()
        self.dataset = normalize(self.dataset)


def imputer(series):
    is_numeric = series.dtype.kind in 'bifc'
    has_na = na_rate(series) > .0
    
    if is_numeric and has_na:
        median = series.median()
        return series.fillna(median)            
    else:
        if has_na:
            mod = self.dataset[col].mode()[0]
            return series.fillna(mod)
    return series


def is_sequential_like_index(series):
    is_numeric = series.dtype.kind in 'bifc'
    if is_numeric:
        first_lower_than_all = (series < series).sum()
        first_diff_is_one = (series[1] - series[0]) == 1
        if not first_lower_than_all and first_diff_is_one:
            return True

    return False


def is_na_ratio_up_thresh(series, threshold):    
    if na_rate(series) > threshold:
        return True

    return False


def is_categorical(series, threshold):
    is_numeric = series.dtype.kind in 'bifc'
    below_thresh = unique_threshold(series, threshold)
    if is_numeric and not below_thresh:
        return False

    return True


def is_categorical_up_thresh(series, threshold):
    below_thresh = unique_threshold(series, threshold)
    return not below_thresh


def na_rate(series):
    return series.isna().sum()/series.shape[0]


def unique_threshold(series, threshold=.1):
    return len(series.unique())/len(series) < threshold


def normalize(df):
    X = df.values #returns a numpy array
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled)

def min_max(x):
    min_ = x.min()
    return (x - min_) / (x.max() - min_)

def main():
    #alphabet = 'abcdefghijklmnopqrstuyxwz'
    #abc = [1,2,3,4,5]
    #cat = [abc[i%5] for i in range(100)]
    #num = np.random.rand(98).tolist() + [np.nan, np.nan]

    #df = pd.DataFrame(list(zip(cat,num,num)))

    df = pd.read_csv('/media/romulo/C4B4FA64B4FA57FE//bases//transplant.csv')

    print(f'Before preparation:\n{df.tail()}\n')

    prepdata = PrepareDataset()

    df = prepdata.fit_transform(df.iloc[:,:-1], df.iloc[:,-1])
    
    print(f'After preparation:\n{df.tail()}')


if __name__ == '__main__':
    main()
