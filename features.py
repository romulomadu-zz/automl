
import networkx as nx
import numpy as np
import statsmodels.api as sm

from random import uniform, seed
from scipy.stats.stats import pearsonr, spearmanr

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



if __name__ == "__main__":
    main()
