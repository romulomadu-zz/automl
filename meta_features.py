
import networkx as nx
import numpy as np
import statsmodels.api as sm

from random import uniform, seed
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.neighbors import KDTree
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from utils import _pprint

__author__ = 'Rômulo Rodrigues <romulomadu@gmail.com>'
__version__ = '0.1'


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
    
    def __init__(self, dataset_name='None', random_state=0):        
        self.random_state = random_state
        self.params_ = {}
        self.dataset_name = dataset_name
            
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
        model = sm.OLS(y, X).fit()
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        # Feed and Calculate indicators
        params_dict = {
            '_dataset_name' : self.dataset_name,
            'c1': c1(X, y),
            'c2': c2(X, y),
            'c3': c3(X, y),
            'c4': c4(X, y),
            'l1': l1(X, y, model),
            'l2': l2(X, y, model),
            'l3': l3(X, y, model),
            's1': s1(y, dist_matrix),
            's2': s2(X, y),
            's3': s3(X, y, dist_matrix),
            's4': s4(X, y),
            't2': t2(X),
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
    return sum(corr_list)/X.shape[1]
    

def c3(X, y):
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

    # Calculate rho of Spearman
    def rho_spearman(d):
        n = d.shape[0]        
        return 1-6*(d**2).sum()/(n**3-n) 
    
    for col in range(ncol):        
        # Calculate rank vectors to Spearman correlation
        rank_x = np.argsort(X[:, col])
        rank_y = np.argsort(y)
        rank_dif = rank_x - rank_y
        
        if rho_spearman(rank_dif) < 0:
            rank_y = rank_y[::-1]
            rank_dif = rank_x-rank_y            

        while abs(rho_spearman(rank_dif)) <= 0.9:
            id_r = np.ndarray.argmax(abs(rank_dif))
            
            for id_j in range(rank_dif.shape[0]):                
                if rank_x[id_j] > rank_x[id_r] and rank_y[id_j] < rank_y[id_r]:
                    rank_dif[id_j] = rank_dif[id_j]-1
                if rank_y[id_j] > rank_y[id_r] and rank_x[id_j] < rank_x[id_r]:
                    rank_dif[id_j] = rank_dif[id_j]+1
            
            rank_dif = np.delete(rank_dif, id_r)
            rank_x = np.delete(rank_x, id_r)
            rank_y = np.delete(rank_y, id_r)
        
        n_j.append(rank_dif.shape[0])
        
    return min(n_j)/n


def c4(X, y, min_resid=0.1):
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

    A = [X.shape[1]-i for i in range(X.shape[1])]
    n = X.shape[0]
       
    while A and X.any():        
        m = np.argmax([abs(spearmanr(X[:, j], y)[0]) for j in range(X.shape[1])])
        A.pop()        
        id_remove = abs(sm.OLS(y, X[:, m]).fit().resid) > min_resid
        X = X[id_remove]
        y = y[id_remove]
      
    return X.shape[0]/n      


def s1(y, dist):
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

    G = nx.from_numpy_matrix(dist)
    T = nx.minimum_spanning_tree(G)
    edges = T.edges()
    edges_dist_norm = min_max(np.array([abs(y[i] - y[j]) for i, j in edges]))

    return edges_dist_norm.sum()/len(edges)
        

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

    X_y = np.hstack((X,y.reshape(X.shape[0],1)))
    X = X_y[X_y[:,-1].argsort()][:, :-1]
    n = X.shape[0]    
    i = 1    
    d = list()

    while i < n:
        d.append(np.linalg.norm(X[i,:]-X[i-1,:]))
        i = i + 1

    return min_max(np.array(d)).sum()/(n-1)


def s3(X, y, dist_matrix):
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
        Normalized 1-NN absolute error
    """

    n = X.shape[0]    
    e = list()
     
    for i in range(n):
        i_nn = np.argmin(np.delete(dist_matrix[i, :], i))
        e.append(abs(y[i]-y[i_nn]))
    
    return min_max(np.array(e)).sum()/n


def s4(X, y, random_state=0):
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
        Normalized 1-NN absolute error
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
    abs_error = np.array([abs(y[int(nearest_ind[i])]-y_[i]) for i in range(y_.shape[0])])
    abs_error_norm = min_max(abs_error)

    return abs_error_norm.sum()/n


def l1(X, y, model):
    """
    Calculate the mean absolute error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
       Ordinary least square model between X,y

    Return
    ------
    float:
        Normalized mean absolute error
    """
    res = model.resid

    return np.mean(res)   


def l2(X, y, model):
    """
    Calculate the mean squared error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
       Ordinary least square model between X,y

    Return
    ------
    float:
        Normalized mean squared error
    """

    n = X.shape[0]
    res = model.resid
    # Normalize squared residuous
    res_norm = min_max(res**2)
    
    return res_norm.sum()/n


def l3(X, y, model, random_state=0):
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
        Normalized mean squared error
    """

    seed(random_state)    
    n = X.shape[0]    
    X_y = np.hstack((X, y.reshape(X.shape[0], 1)))
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

    res_norm = min_max(abs(model.predict(X_) - y_.T))

    return res_norm.sum()/n


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

    return X.shape[0]/X.shape[1]


def min_max(x):
    min_ = x.min()
    return (x-min_)/(x.max()-min_)


def main():
    boston = load_boston()
    X = boston["data"]
    y = boston["target"]
    mf = MetaFeatures(dataset_name='Boston')

    print(mf.fit(X, y))


if __name__ == "__main__":
    main()

