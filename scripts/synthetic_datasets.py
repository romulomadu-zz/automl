# coding=utf-8

import numpy as np
import pandas as pd

np.random.seed(0)


def generate_poly(filename, degree=1, n=500, m=1, noise_variance=0):
	X = np.random.rand(n,m)
	beta = np.random.rand(degree+1, 1)

	list_y = list()
	for i in range(n):
		list_x =list()
		for p in range(degree+1):
			list_x.append(X[i,:] ** p)
		y_i = np.array(list_x).T.dot(beta).sum() + np.random.normal(0, noise_variance)
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

	path = '/media/romulo/C4B4FA64B4FA57FE//datasets_prep//'
	n = 500
	n_features = [1, 2, 5, 10]
	n_poly_degree = [1, 3, 5]
	n_std = [.0, .1, 0.2]
	n_freq = [1, 3]

	for m in n_features:
		for d in n_poly_degree:
			for s in n_std:
				name = f'{path}poly_{d}_{m}_{s}.csv'
				print(f'Created at {name}')
				generate_poly(name, degree=d, m=m, noise_variance=s)

	for m in n_features:
		for w in n_freq:
			for s in n_std:
				name = f'{path}sin_{w}_{m}_{s}.csv'
				print(f'Created at {name}')
				generate_sin(name, w=w, m=m, noise_variance=s)