import pandas as pd
import re

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import (check_is_fitted, check_array, FLOAT_DTYPES)
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.pipeline import make_pipeline


class RemoveNaColumns(TransformerMixin):
    """
    Transformer to remove columns from dataset which the number of NaN values in proporrtion of database size exceeds a passed value between 0 and 1.
    """

    def __init__(self, na_proportion=.2):
        self.na_proportion = na_proportion

    def fit(self, X, y=None):
        n_cols = X.shape[1]
        
        cols = []
        for i in range(n_cols):
            if self.get_na_proportion(X.iloc[:, i]) < self.na_proportion:
                cols.append(i)

        self.cols = cols
        return self

    def transform(self, X):
        return X.iloc[:, self.cols]


    def get_na_proportion(self, col):
        return col.isna().sum() / col.shape[0]


class RemoveCategorical(TransformerMixin):
    """
    Transformer to remove columns from dataset which the number of Categories in proporrtion of database size exceeds a passed value between 0 and 1.
    """

    def __init__(self, cat_proportion=.02):
        self.cat_proportion = cat_proportion
        
    def fit(self, X, y=None):
        n_cols = X.shape[1]

        cols = []
        for i in range(n_cols):
            col = X.iloc[:, i]
            up_thresh = self.get_cat_proportion(col) > self.cat_proportion
            if self.is_categorical(col) and up_thresh:
                continue
            cols.append(i)		
        self.cols = cols

        return self

    def transform(self, X):
        return X.iloc[:, self.cols]

    def is_numeric(self, col):
        return col.dtype.kind in 'bifc'

    def is_categorical(self, col):
        up_thresh = self.get_cat_proportion(col) > self.cat_proportion
        return not (self.is_numeric(col) and up_thresh)

    def get_cat_proportion(self, col):
        return len(col.unique()) / col.shape[0]
    

class RemoveSequential(TransformerMixin):
    """
    Transformer to remove columns from dataset which have sequential values 
    """

    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        n_cols = X.shape[1]

        cols = []
        for i in range(n_cols):
            col = X.iloc[:, i]
            
            if self.is_sequential(col):
                continue
            cols.append(i)
        self.cols = cols

        return self

    def transform(self, X):
        return X.iloc[:, self.cols]
    
    def is_sequential(self, series):
        is_numeric = series.dtype.kind in 'bifc'
        if is_numeric:
            first_lower_than_all = (series < series).sum()
            first_diff_is_one = (series[1] - series[0]) == 1
            if not first_lower_than_all and first_diff_is_one:
                return True

        return False       


class ImputerByColumn(RemoveCategorical):
    """
    Impute values given a strategy, if column is categorical, it wil use most frequent value 
    """

    def __init__(self, missing_values='NaN', strategy='mean', cat_proportion=.02):
        super().__init__()
        self.strategy = strategy
        self.missing_values = missing_values
        #self.cat_proportion = cat_proportion
        
    def fit(self, X, y=None):
        n_cols = X.shape[1]

        values = []
        for i in range(n_cols):
            col = X.iloc[:, i]
            
            if col.isna().sum():                
                if self.is_categorical(col):
                    values.append(col.mode()[0])
                else:
                    if self.strategy == 'mean':
                        values.append(col.mean())
                    if self.strategy == 'median':
                        values.append(col.median())
            else:
                values.append(None)

        self.values = values

        return self
        
    def transform(self, X):
        for i in range(X.shape[1]):
            if self.values[i]:
                X.iloc[:, i] = X.iloc[:, i].fillna(self.values[i])
        
        return X


class DFOneHotEncoder(RemoveCategorical):
    """
    Wraps sklearn's OneHotEncoder class to use pandas DataFrame.
    """

    def __init__(self, cat_proportion=.02):
        super().__init__() 
        
    def fit(self, X, y=None):
        n_cols = X.shape[1]

        encoders = []
        for i in range(n_cols):
            col = X.iloc[:, i]
            
            if self.is_categorical(col):
                encoders.append(LabelBinarizer().fit(col.astype(str)))
            else:
                encoders.append(None)

        self.encoders = encoders

        return self
        
    def transform(self, X):
        series_list = []
        for i in range(X.shape[1]):
            col = X.iloc[:, i]
            
            if self.encoders[i]:
                cols = self.encoders[i].transform(col.astype(str))
                if cols.shape[1] == 1:
                    series_list.append(pd.Series(cols.flatten(), name=col.name))
                else:
                    df = pd.DataFrame(cols, columns=list(map(lambda x: f'{i}_{x}', self.encoders[i].classes_)))
                    series_list.append(df)
            else:
                series_list.append(col)

        return pd.concat(series_list, axis=1)


class DFMinMaxScaler(MinMaxScaler):    
    """
    Wraps sklearn's MinMaxScaler class to use pandas DataFrame.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__()
        
    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """

        # Reset internal state before fitting
        self.classes_ = X.columns
        self._reset()
        return self.partial_fit(X, y)
        
    def transform(self, X):
        """Scaling features of X according to feature_range.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X *= self.scale_
        X += self.min_
        return pd.DataFrame(X, columns=self.classes_)


def process_file(file_path, cat_proportion=.05, na_proportion=.1):
    # For files downloaded from OpenML
    with open(file_path) as f:
        data = f.read()
    with open(file_path, 'w') as f:
        f.write(re.sub('{|}', '', data))

    try:
        dataset = pd.read_csv(file_path).replace(
            {'?': np.nan}).apply(pd.to_numeric, errors='ignore')
    except:
        dataset = pd.read_csv(file_path)

    def clean_str(x):
        try:
            return float(x.strip().split(' ')[-1])
        except:
            return x

    dataset = dataset.applymap(clean_str)

    pipe = make_pipeline(
                 RemoveNaColumns(na_proportion=na_proportion), 
                 RemoveCategorical(cat_proportion=cat_proportion), 
                 RemoveSequential(), 
                 ImputerByColumn(cat_proportion=cat_proportion),
                 DFOneHotEncoder(cat_proportion=cat_proportion),
                 DFMinMaxScaler()                
                )
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    #dataset_out = prepdata.fit_transform(X, y)
    return pipe.fit_transform(X, y)


if __name__ == '__main__':
    pipe = make_pipeline(
                     RemoveNaColumns(na_proportion=.1), 
                     RemoveCategorical(cat_proportion=.05), 
                     RemoveSequential(), 
                     ImputerByColumn(cat_proportion=.05),
                     DFOneHotEncoder(cat_proportion=.05),
                     DFMinMaxScaler()                
                    )
    df = pd.read_csv('/media/romulo/C4B4FA64B4FA57FE//datasets//dataset_2190_cholesterol.csv')
    print(f'Before preparation:\n{df.tail()}\n')

    df = pipe.fit_transform(df.iloc[:,:-1], df.iloc[:,-1])
    print(f'After preparation:\n{df.tail()}')



