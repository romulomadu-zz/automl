# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_poly(filename, degree=1, n=500, m=1, noise_variance=0):
	X = np.random.rand(n,m)
	roots = (np.random.rand(m, degree) * 2) - 1
	#beta0 = (np.random.rand() * 2) - 1

	list_y = list()
	for i in range(n):
		res = 0
		for d in range(m):
			mul = 1
			for p in range(degree):
				mul *= (X[i,d] + roots[d][p])
			res += mul
		list_y.append(res)
	y = np.array(list_y)
	y = MinMaxScaler().fit_transform(y.reshape(n, 1))

	dataset = np.concatenate([X, y], axis=1)
	pd.DataFrame(dataset, columns=list(range(m+1))).to_csv(filename)


def generate_sin(filename, c=1, w=1, n=500, m=1, noise_variance=0):
	X = np.random.rand(n,m)
	phase = np.random.uniform(0, np.pi, (1,m))

	list_y = list()
	for i in range(n):
		y_i = c * np.sin(2 * np.pi * w * X[i,:] + phase).sum() + np.random.normal(0, noise_variance)
		list_y.append(y_i)
	dataset = np.concatenate([X, np.array(list_y).reshape(n, 1)], axis=1)
	pd.DataFrame(dataset, columns=list(range(m+1))).to_csv(filename)

if __name__ == '__main__':

	path = '//home//romulo//TCC-PEDS//datasets_synthetic//'
	n = 500
	n_features = [1, 2, 5, 10]
	n_poly_degree = [1, 3, 5]
	n_std = [.0, .5, 1.0]
	n_freq = [1, 3]
	n_variants = 50

	for v in range(n_variants):
		for m in n_features:
			for d in n_poly_degree:
				for s in n_std:
					name = '{:}poly_{:}_{:}_{:}_{:}.csv'.format(path, d, m, s, v)
					print('Created at {:}'.format(name))
					generate_poly(name, degree=d, m=m, noise_variance=s)

	for v in range(n_variants):
		for m in n_features:
			for w in n_freq:
				for s in n_std:
					name = '{:}sin_{:}_{:}_{:}_{:}.csv'.format(path, w, m, s, v)
					print('Created at {:}'.format(name))
					generate_sin(name, w=w, m=m, noise_variance=s)