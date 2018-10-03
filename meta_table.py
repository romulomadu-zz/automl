
#import statsmodels.api as sm

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from meta_features import *
from utils import _pprint
import six

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
    
    def __init__(self, random_state=0):        
        self.random_state = random_state
        self.params_ = {}
            
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


def main():
    boston = load_boston()
    X = boston["data"]
    y = boston["target"]
    mf = MetaFeatures()

    print(mf.fit(X, y).get_params())


if __name__ == "__main__":
    main()
