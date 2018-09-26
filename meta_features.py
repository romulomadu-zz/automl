
import statsmodels.api as sm
import six

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from features import *


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
