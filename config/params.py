from numpy import zeros

from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import uniform as sp_uniform
from skopt.space import Real


def nmse(y_pred, y_true, verbose=0):
	"""	Normalized mean squared error."""
	n = y_pred.shape[0]
	error = mean_squared_error(y_true, y_pred) / mean_squared_error(y_pred, zeros(n))
	if verbose:
		print(f'NMSE: {nmse_}')
	return error


def grid_params():
	"""Grid Search parameters."""
	gamma = [2 ** i for i in range(-15, 4, 1)]
	C = [2 ** i for i in range(-5, 16, 1)]
	epsilon = [2 ** i for i in range(-8, 2, 1)]

	param_grid = {
		'param_grid': {'kernel': ['rbf'], 'C': C, 'gamma': gamma, 'epsilon': epsilon},
		'scoring': make_scorer(nmse, greater_is_better=False),
		'verbose': 1,
		'cv': 10
	}

	return param_grid


def random_params():
	"""Random Search parameters."""
	gamma = sp_uniform(2 ** -15, 2 ** 3)
	C = sp_uniform(2 ** -5, 2 ** 15)
	epsilon = sp_uniform(2 ** -8, 2 ** 1)

	param_dist = {
		'param_distributions': {'kernel': ['rbf'], 'C': C, 'gamma': gamma, 'epsilon': epsilon},
		'scoring': make_scorer(nmse, greater_is_better=False),
		'verbose': 1,
		'n_iter': 1000,
		'cv': 10
	}

	return param_dist


def bayes_params():
	"""Random Search parameters."""
	gamma = Real(2 ** -15, 2 ** 3)
	C = Real(2 ** -5, 2 ** 15)
	epsilon = Real(2 ** -8, 2 ** 1)

	search_spaces = {
		'search_spaces': {'kernel': ['rbf'], 'C': C, 'gamma': gamma, 'epsilon': epsilon},
		'scoring': make_scorer(nmse, greater_is_better=False),
		'verbose': 1,
		'n_iter': 1000,
		'cv': 10
	}

	return search_spaces


