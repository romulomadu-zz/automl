# coding=utf-8

import numpy as np
import pandas as pd


def generate_poly(filename, degree_list=[1], n=500, m=1, noise_variance=0):
	X = np.random.rand(n,m)
	beta = np.random.rand(m, len(degree_list) + 1)

	list_y = list()
	for i in range(n):
		list_x =list()
		for j in range(m):
			list_x_j = list()
			for p in degree_list:
				list_x_j.append(X[i,j] ** p)			
			list_x.append([1] + list_x_j)
		y_i = (np.array(list_x) * beta).sum() + np.random.normal(0, noise_variance)
		list_y.append(y_i)
	dataset = np.concatenate([X, np.array(list_y).reshape(n, 1)], axis=1)
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
			for d in range(len(n_poly_degree)):
				for s in n_std:
					name = '{:}poly_{:}_{:}_{:}_{:}.csv'.format(path, n_poly_degree[d], m, s, v)
					print('Created at {:}'.format(name))
					generate_poly(name, degree_list=n_poly_degree[:d+1], m=m, noise_variance=s)

	for v in range(n_variants):
		for m in n_features:
			for w in n_freq:
				for s in n_std:
					name = '{:}sin_{:}_{:}_{:}_{:}.csv'.format(path, w, m, s, v)
					print('Created at {:}'.format(name))
					generate_sin(name, w=w, m=m, noise_variance=s)