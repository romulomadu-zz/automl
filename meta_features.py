import numpy as np
import statsmodels.api as sm
import operator
import six
import networkx as nx

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from random import uniform, seed


def _pprint(params, offset=0, printer=repr):
    """
    Pretty print the dictionary 'params'.
    
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):        
        if type(v) is float:

            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:

            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
                
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)

    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))

    return lines


class BaseMeta(object):    
    
    def get_params(self):
        pass
    
    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(),
                                               offset=len(class_name),),)
    

class MetaFeatures(BaseMeta):
    
    def __init__(self, random_state=0):        
        self.random_state = random_state
        self.params_ = {}
        
            
    def fit(self, X, y):
        # Pre calculate some indicators inputs
        X = MinMaxScaler().fit_transform(X)       
        model = sm.OLS(y, X).fit()        
        dist_matrix = squareform(pdist(X, metric='euclidean'))        
        # Feed and Calculate indicators
        self.params_ = {
            'c1' : c1(X, y),
            'c2' : c2(X, y),
            'c3' : c3(X, y),
            'c4' : c4(X, y),
            'l1' : l1(X, y, model),
            'l2' : l2(X, y, model),
            'l3' : l3(X, y, model),
            'l4' : l4(X, y),
            's1' : s1(y, dist_matrix),
            's2' : s2(X, y),
            's3' : s3(X, y, dist_matrix),
            't2' : t2(X),            
        }        
        
        return self
        
    def get_params(self):        
        return self.params_ 

def c1(X, y):
    corrList = list()
    for col in range(X.shape[1]):
        corr = spearmanr(X[:,col],y)
        corrList.append(abs(corr[0]))
    return max(corrList)

def c2(X, y):
    corrList = list()
    for col in range(X.shape[1]):
        corr = spearmanr(X[:,col],y)
        corrList.append(abs(corr[0]))
    return sum(corrList)/X.shape[1]
    
def c3(X, y):    
    # Initial variables
    ncol = X.shape[1]
    n = X.shape[0]    
    n_j = list()

    # Calculate rho of Spearman
    def rho_spearman(d):
        n = d.shape[0]        
        return 1-6*np.ndarray.sum(d**2)/(n**3-n) 
    
    for col in range(ncol):        
        # Calculate rank vectors to Spearman correlation
        rank_x = np.argsort(X[:,col])
        rank_y = np.argsort(y)
        rank_dif = rank_x - rank_y
        
        if (rho_spearman(rank_dif)<0):
            rank_y = rank_y[::-1]
            rank_dif = rank_x-rank_y            

        while abs(rho_spearman(rank_dif))<=0.9:
            id_r = np.ndarray.argmax(abs(rank_dif))
            
            for id_j in range(rank_dif.shape[0]):                
                if rank_x[id_j]>rank_x[id_r] and rank_y[id_j]<rank_y[id_r]:
                    rank_dif[id_j] = rank_dif[id_j]-1
                if rank_y[id_j]>rank_y[id_r] and rank_x[id_j]<rank_x[id_r]:
                    rank_dif[id_j] = rank_dif[id_j]+1
            
            rank_dif = np.delete(rank_dif, id_r)
            rank_x = np.delete(rank_x, id_r)
            rank_y = np.delete(rank_y, id_r)
        
        n_j.append(rank_dif.shape[0])
        
    return min(n_j)/n

def c4(X, y, minResid=0.1):    
    A = [X.shape[1]-i for i in range(X.shape[1])]
    n = X.shape[0]
       
    while A and X.any():        
        m = np.argmax([abs(spearmanr(X[:,j],y)[0]) for j in range(X.shape[1])])        
        A.pop()        
        id_remove = abs(sm.OLS(y,X[:,m]).fit().resid) > minResid        
        X = X[id_remove]
        y = y[id_remove]
      
    return X.shape[0]/n      

def s1(y, dist):
    G = nx.from_numpy_matrix(dist)
    T = nx.minimum_spanning_tree(G)
    edges = T.edges()    
    return sum([abs(y[i] - y[j]) for i, j in edges])/len(edges)
        

def s2(X, y):    
    X_y = np.hstack((X,y.reshape(X.shape[0],1)))
    X = X_y[X_y[:,-1].argsort()][:,:-1]    
    n = X.shape[0]    
    i = 1    
    d = 0
    
    while i < n:        
        d = d + np.linalg.norm(X[i,:]-X[i-1,:])
        i = i + 1
        
    return d/n

def s3(X, y, dist_matrix):
    n = X.shape[0]    
    e = 0
     
    for i in range(n):
        i_nn = np.argmin(np.delete(dist_matrix[i,:],i))
        e = e + (y[i]-y[i_nn])**2
    
    return e/n

def S4(X,y):
    pass

def l1(X, y, model):   
    n = X.shape[0]
    res = model.resid   
    
    return np.mean(res)   

def l2(X, y, model):    
    n = X.shape[0]
    res = model.resid
    
    return sum(res)**2/(n-1)

def l3(X, y, model, random_state=0):    
    seed(random_state)    
    n = X.shape[0]    
    X_y = np.hstack((X,y.reshape(X.shape[0],1)))
    X = X_y[X_y[:,-1].argsort()][:,:-1]
    y = sorted(y)    
    i = 0    
    X_ = X[0,:].reshape(1,X.shape[1])
    y_ = y[0].reshape(1,1)
    
    while i < n-2:        
        x_i = np.array([uniform(X[i,j], X[i-1,j]) for j in range(X.shape[1])])
        X_ = np.vstack((X_,x_i.reshape(1,X.shape[1])))        
        y_i = np.array([uniform(y[i], y[i-1])])
        y_ = np.vstack((y_,y_i.reshape(1,1)))        
        i = i + 1
        
    return np.ndarray.sum((model.predict(X_)-y_)**2)/(n)    

def l4(X, y):
    pass

def t2(X):
    return X.shape[0]/X.shape[1]


def main():
    boston = load_boston()
    X = boston["data"]
    y = boston["target"]
    names = boston["feature_names"]
    mf = MetaFeatures()
    res = mf.fit(X, y)
    print(res)


if __name__ == "__main__":
    main()


