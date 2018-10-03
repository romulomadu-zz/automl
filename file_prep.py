import re
import pandas as pd
import numpy as np

from glob import glob
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class PrepDataset(object):

    def __init__(self, n_unique_values_treshold=15):
        self.n_unique_values_treshold = n_unique_values_treshold

    def fit_transform(dataset):
        return self.imputer(self._remove_categorical(df))

    def _determine_type_of_feature(self, df):        
        feature_types = []

        for feature in df.columns:
            if feature != 'label':
                unique_values = df[feature].unique()
                example_value = unique_values[0]

                if (isinstance(example_value, str)) or (len(unique_values) <= self.n_unique_values_treshold):
                    feature_types.append('categorical')
                else:
                    feature_types.append('continuous')
        
        return feature_types

    def _remove_categorical(self, df):
        label_data = df.iloc[:,-1]
        features_data = df.iloc[:,:-1]
        feature_types = self._determine_type_of_feature(features_data)
        n_unique_values_treshold = 15
        
        for col in features_data.columns[feature_types=='categorical']:
            if len(features_data[col].unique()) > self.n_unique_values_treshold:
                features_data = features_data.drop(col, axis=1)

        return pd.concat(features_data, label_data, axis=1)

    @staticmethod
    def _nan_counts(series):
        return series.isna().sum()/series.shape[0]

    def _imputer(self, df):
        df_types = self._determine_type_of_feature(df)
        
        for col, typ in zip(df.columns, df_types):
            if _nan_counts(df[col]) > 0.1:
                if typ == 'continuous':
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna(None)
                continue
            if typ == 'continuous':
                df[col] = df[col].fillna(df[col].median())
                continue
            if typ == 'categorical':
                df[col] = df[col].fillna(df[col].mode())

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

    prepdata = PrepDataset()

    df = prepdata.fit_transform(df)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    rf = RandomForestRegressor(random_state=0)
    model = rf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(mean_squared_error(y_test, y_pred))

    import matplotlib.pyplot as plt

    plt.scatter(y_test, y_pred)

if __name__ == '__main__':
    main()