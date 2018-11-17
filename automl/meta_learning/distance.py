import pandas as pd
import numpy as np

def l1_norm_unweighted(x, metabase, k=3):
	n, m =  metabase.shape

	def dist(x, x_i, metabase):
		numerator = abs(x - x_i)
		denominator = np.max(metabase, axis=0) - np.min(metabase, axis=0)

		return (numerator / denominator).sum()

	dist_list = list()
	for i in range(n):
		dist_list.append(dist(x, metabase[i, :], metabase))

	return np.array(dist_list).argsort()[:k]


if __name__ == '__main__':
	x = np.array([1, 2])
	metabase = np.array([
		[3, 1],
		[1, 2],
		[4, 3]
		])

	print(l1_norm_unweighted(x, metabase))




