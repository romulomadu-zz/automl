    
# coding=utf-8

import multiprocessing
import networkx as nx
import numpy as np

from random import uniform, seed, randint
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.neighbors import KDTree
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from joblib import Parallel, delayed

try:
    from automl.utils import _pprint
except:
    import sys
    sys.path.append("..")
    from utils import _pprint

import warnings
warnings.filterwarnings("ignore")


__author__ = 'Romulo Rodrigues <romulomadu@gmail.com>'
__version__ = '0.1.0'


class BaseMeta(object):
    """
    Base class for meta features evaluators objects.
    """
    
    def get_params(self):
        pass
    
    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(),
                                               offset=len(class_name),),)


class MetaFeatures(BaseMeta):
    """
    Meta feature evaluator for regression problems.
    """
    
    def __init__(self, dataset_name='None', random_state=0, metric='mse'):
        self.random_state = random_state
        self.dataset_name = dataset_name
        self.metric = metric
            
    def fit(self, X, y):
        self.params_ = self._calculate_meta_features(X,y)
        return self
        
    def get_params(self):        
        return self.params_

    def _calculate_meta_features(self, X, y):
        """
        Calculate meta features.

        Parameters
        ----------
        X : numpy.array
            2D array with features columns
        y : numpy.array
            Array of true values

        Return
        ------
        object :
            MetaFeatures object with params calculated

        """

        # Pre calculate some indicators inputs
        X = MinMaxScaler().fit_transform(X)
        model = LinearRegression().fit(X, y)
        svr = SVR(kernel='linear').fit(X, y)
        resid = y - model.predict(X)
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        # Feed and Calculate indicators
        params_dict = {
            'dataset' : self.dataset_name,
            'm_c1': c1(X, y),
            'm_c2': c2(X, y),
            'm_c3': c3(X, y, n_jobs=32),
            'm_c4': c4(X, y, n_jobs=32),
            'm_l1': l1(X, y, resid),
            'm_l2': l2(X, y, resid),
            'm_l3': l3(X, y, svr, metric=self.metric),
            'm_s1': s1(y, dist_matrix),
            'm_s2': s2(X, y),
            'm_s3': s3(X, y, dist_matrix, self.metric),
            'm_s4': s4(X, y, metric=self.metric),
            'm_t2': t2(X)
        }

        return params_dict


def c1(X, y):
    """
    Calculate the maximum feature correlation to the output.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values

    Return
    ------
    float:
        Maximum feature correlation to the output

    """

    corr_list = list()
    for col in range(X.shape[1]):
        corr = spearmanr(X[:, col], y)
        corr_list.append(abs(corr[0]))
    return max(corr_list)


def c2(X, y):
    """
    Calculate the average feature correlation to the output.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values

    Return
    ------
    float:
        Average feature correlation to the output
    """

    corr_list = list()
    for col in range(X.shape[1]):
        corr = spearmanr(X[:, col], y)
        corr_list.append(abs(corr[0]))
    return np.mean(corr_list)
    

def c3(X, y, n_jobs=1):
    """
    Calculate the individual feature efficiency.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values

    Return
    ------
    float:
        minimum features elements droped to achieve correlation of 0.9 divided by n
    """

    # Initial variables
    ncol = X.shape[1]
    n = X.shape[0]    
    n_j = list()
    
    rank_all_y = rankdata(y)
    rank_all_y_inv = rank_all_y[::-1]
    
    num_cores = multiprocessing.cpu_count()
    if num_cores < n_jobs:
        n_jobs = num_cores
    n_j = Parallel(n_jobs=n_jobs)(delayed(removeCorrId)(X[:,col], rank_all_y, rank_all_y_inv) for col in range(ncol))
        
    return min(-np.array(n_j) + n) / n


def c4(X, y, min_resid=0.1, n_jobs=1):
    """
    Calculate the collective feature efficiency.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values

    Return
    ------
    float:
        Ratio between number of points that put residuos lower than 0.1
        and total number fo points.
    """

    A = list(range(X.shape[1]))
    n = X.shape[0]
    mcol = X.shape[1]
    num_cores = multiprocessing.cpu_count()
    if num_cores < n_jobs:
        n_jobs = num_cores
       
    while A and X.any():
        pos_rho_list = Parallel(n_jobs=n_jobs)(delayed(calculateCorr)(X[:, j], y, j, A) for j in range(mcol))
        rho_list = [t[1] for t in sorted(pos_rho_list)]
        
        if sum(rho_list) == .0:
            break                    
        m = np.ndarray.argmax(np.array(rho_list))
        A.remove(m)
        model = LinearRegression()
        x_j = X[:, m].reshape((-1, 1))
        y = y.reshape((-1, 1))
        model.fit(x_j, y)

        resid = y - model.predict(x_j)
        id_remove = abs(resid.flatten()) > min_resid
        X = X[id_remove, :]
        y = y[id_remove]
      
    return len(y) / n      


def s1(y, dist_matrix):
    """
    Calculate the output distribution.

    Parameters
    ----------
    y : numpy.array
        Array of response values

    Return
    ------
    float:
        Normalized output distribution mean value
    """

    G = nx.from_numpy_matrix(dist_matrix)
    T = nx.minimum_spanning_tree(G)
    edges = T.edges()
    edges_dist_norm = np.array([abs(y[i] - y[j]) for i, j in edges])

    return edges_dist_norm.sum() / len(edges)
        

def s2(X, y):
    """
    Calculate the input distribution.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values

    Return
    ------
    float:
        Normalized input distribution mean value
    """

    X_y = np.hstack((X,y.reshape(X.shape[0], 1)))
    X = X_y[X_y[:, -1].argsort()][:, :-1]
    n = X.shape[0]    
    i = 1    
    d = list()

    while i < n:
        d.append(np.linalg.norm(X[i, :]-X[i-1, :]))
        i = i + 1

    return np.array(d).sum() / (n - 2)


def s3(X, y, dist_matrix, metric='mse'):
    """
    Calculate the error of the nearest neighbor regressor.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    dist_matrix : numpy.array
        2d-array with euclidean distances between features

    Return
    ------
    float:
        Normalized 1-NN mean error
    """

    n = X.shape[0]    
    error = list()
     
    for i in range(n):
        i_nn = np.argmin(np.delete(dist_matrix[i, :], i))
        # Add 1 to i_nn in case equals to i
        if i==i_nn:
            i_nn = i_nn + 1
        error.append(y[i]-y[i_nn])

    return compute_metric(np.array(error), metric)


def s4(X, y, random_state=0, metric='mse'):
    """
    Calculate the non-linearity of nearest neighbor regressor

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    random_state : int
        Seed to random calculations

    Return
    ------
    float:
        Normalized 1-NN error
    """

    seed(random_state)  
    tree = KDTree(X)
    n = X.shape[0]    
    X_y = np.hstack((X,y.reshape(X.shape[0], 1)))
    X = X_y[X_y[:, -1].argsort()][:, :-1]
    y = sorted(y)    
    i = 1    
    X_ = X[0, :].reshape(1, X.shape[1])
    y_ = y[0].reshape(1, 1)
    
    while i < n:        
        x_i = np.array([uniform(X[i, j], X[i-1, j]) for j in range(X.shape[1])])
        X_ = np.vstack((X_, x_i.reshape(1, X.shape[1])))
        y_i = np.array([uniform(y[i], y[i-1])])
        y_ = np.vstack((y_, y_i.reshape(1, 1)))
        i = i + 1

    nearest_dist, nearest_ind = tree.query(X_, k=1)
    error = np.array([y[int(nearest_ind[i])]-y_[i] for i in range(y_.shape[0])])

    return compute_metric(error, metric)


def l1(X, y, resid):
    """
    Calculate the mean absolute error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model :np.array
       Linear regression model residuals between X,y

    Return
    ------
    float:
        Mean absolute error
    """
    return np.mean(abs(resid))


def l2(X, y, resid):
    """
    Calculate the mean squared error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model :np.array
       Linear regression model residuals between X,y
    Return
    ------
    float:
        Mean squared error
    """

    # Normalize squared residuous
    res_norm = resid ** 2
    
    return np.mean(res_norm)


def l3(X, y, model, random_state=0, metric='mse'):
    """
    Calculate the non-linearity of a linear regressor

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
       Ordinary least square model between X,y
    random_state : int
        Seed to random calculations

    Return
    ------
    float:
        Normalized mean error
    """

    seed(random_state) 
    n, m = X.shape
    y = y.flatten()
    idx_sorted_y = y.argsort()
    X_sorted = X[idx_sorted_y, :]
    y_sorted  = y[idx_sorted_y]
    i = 1
    X_list = list()
    y_list = list()

    while i < n:        
        x_i_list = list()
        for j in range(m):
            uniques_values = np.unique(X_sorted[:, j])
            if len(uniques_values) <= 2:
                x_i_list.append(randint(0, 1))
            else:
                x_i_list.append(uniform(X_sorted[i, j], X_sorted[i-1, j]))
        x_i = np.array(x_i_list)
        y_i = np.array([uniform(y_sorted[i], y_sorted[i-1])])
        
        X_list.append(x_i)
        y_list.append(y_i)
        i = i + 1

    X_ = np.array(X_list)
    y_ = np.array(y_list)   
    error = model.predict(X_).reshape((n-1,)) - np.array(y_).reshape((n-1,))

    return compute_metric(error, metric)


def t2(X):
    """
    Calculate the average number of examples per dimension.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns

    Return
    ------
    float:
        Ratio between number of examples and number of features
    """

    return X.shape[0] / X.shape[1]


def compute_metric(arr, metric):
    """Compute normalized chosen metric."""
    n = max(arr.shape)
    # Select metric to return
    if metric == 'mae':
        return np.abs(np.array(arr)).sum() / n
    if metric == 'mse':
        return (np.array(arr) ** 2).sum() / (n - 1)
    if metric == 'rmse':
        return (np.sqrt(np.array(arr) ** 2)).sum() / n


def min_max(x):
    """Min-max scaler."""    
    min_ = x.min()
    max_ = x.max()
    if min_ == max_:
        return x / min_
    return (x - min_) / (max_- min_)


def rho_spearman(d):
    """Calculate rho of Spearman."""    
    n = d.shape[0]        
    return 1 - 6 * (d**2).sum() / (n**3 - n) 


def removeCorrId(x_j, rank_all_y, rank_all_y_inv):
    """Calculate rank vectors to Spearman correlation."""
    rank_x = rankdata(x_j)
    rank_y = rank_all_y
    rank_dif = rank_x - rank_y
    
    if rho_spearman(rank_dif) < 0:
        rank_y = rank_all_y_inv
        rank_dif = rank_x - rank_y            

    while abs(rho_spearman(rank_dif)) <= .9:
        id_r = np.ndarray.argmax(abs(rank_dif))
        rank_dif = rank_dif + (rank_y > rank_y[id_r]) - (rank_x > rank_x[id_r])            
        rank_dif = np.delete(rank_dif, id_r)
        rank_x = np.delete(rank_x, id_r)
        rank_y = np.delete(rank_y, id_r)

    return len(rank_dif)


def calculateCorr(x_j, y, j, A):
    """Calculate absolute Spearman correlation for x_j in set A."""
    if j in A:
        corr = abs(spearmanr(x_j, y)[0])
    else:
        corr = .0
    
    return (j, corr)    


def main():
    boston = load_boston()
    X = boston["data"]
    y = boston["target"]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_X.fit_transform(y.reshape(-1, 1))

    mf = MetaFeatures(dataset_name='Boston', metric='mse')

    print(mf.fit(X, y))


if __name__ == "__main__":
    main()

