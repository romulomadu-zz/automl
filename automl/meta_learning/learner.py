import numpy as np 

from sklearn.base import BaseEstimator
try:
    from automl.meta_learning import l1_norm_unweighted
except:
    import sys
    sys.path.append("..")
    from meta_learning import l1_norm_unweighted

class KNearestNeighbors(BaseEstimator):

	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.metafeatures = X
		self.metatargets = y

		return self

	def predict(self, x):
		idx = l1_norm_unweighted(x, self.metafeatures, self.k)

		return self.metatargets[idx]


if __name__ == '__main__':
	x = np.array([1, 2])
	X = np.array([
		[3, 1],
		[1, 2],
		[4, 3]
		])

	y = np.array(['a', 'b', 'c'])

	learner = KNearestNeighbors(k=2)
	learner.fit(X, y)
	print(learner.predict(x))



