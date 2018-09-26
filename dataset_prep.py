import re
import pandas as pd
import numpy as np

from glob import glob
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def remove_categorical(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df = df.drop(col, axis=1)
    return df    

def freq_nan(series):
    return series.isna().sum()/series.shape[0]

def imputer(df):
    for col in df.columns:
        if freq_nan(df[col]) > 0.1:
            df[col] = df[col].fillna(.0)
            continue
        df[col] = df[col].fillna(df[col].median())
        
    return df

def main():
    files_list = glob('D://bases//*.csv')

    file_path = files_list[np.random.randint(0, len(files_list)-1)]
    print(file_path.split('\\')[-1])

    with open(file_path) as f:
        data = f.read()
    with open(file_path, 'w') as f:
        f.write(re.sub('{|}','', data))
        

    try:
        df = pd.read_csv(file_path).replace({'?':np.nan}).apply(pd.to_numeric, errors='ignore')
    except:
        df = pd.read_csv(file_path)

    df = remove_categorical(df)

    df = imputer(df)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    rf = RandomForestRegressor(random_state=0)
    model = rf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mean_squared_error(y_test, y_pred)

    import matplotlib.pyplot as plt

    plt.scatter(y_test, y_pred)