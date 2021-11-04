from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from scipy.stats import norm, kurtosis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


def plot_fitness(l_curve, plt_name, title):

	# print(l_curve)
	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Fitness Score")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()

def plot_fitness_vs_fevals(l_curve, plt_name, title):

	# print(l_curve)
	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Function Evals")
	plt.ylabel("Fitness Score")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()


def plot_iters_vs_clock(l_curve, plt_name, title):

	# print(l_curve)
	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Clock Time")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()

def plot_fitness_runner(l_curve, plt_name, title):

	plt.plot(l_curve)
	# plt.legend(["iterations", "validation_data"])
	plt.title(title)
	plt.xlabel("Iterations")
	plt.ylabel("Fitness Score")
	plt.grid()
	plt.savefig(plt_name + '.png',bbox_inches='tight')
	plt.close()

def multi_plot(l_curve, plt_name, title, itr, trials, save_file):

	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	print(trials)
	# plt.legend(["iterations", "validation_data"])
	if save_file:
		leg = []
		for i in range(trials):
			leg.append("Trial " + str(i+1))
		plt.legend(leg)
		plt.title(title)
		plt.xlabel("Iterations")
		plt.ylabel("Fitness Score")
		plt.grid()
		plt.savefig(plt_name + '.png',bbox_inches='tight')
		plt.close()


def multi_plot_fevals(l_curve, plt_name, title, itr, trials, save_file):

	fitness = np.zeros(np.shape(l_curve)[0])
	itr = np.zeros(np.shape(l_curve)[0])
	for i in range(np.shape(l_curve)[0]):
		fitness[i] = l_curve[i][0]
		itr[i] = l_curve[i][1]
	plt.plot(itr, fitness)
	print(trials)
	# plt.legend(["iterations", "validation_data"])
	if save_file:
		leg = []
		for i in range(trials):
			leg.append("Trial " + str(i+1))
		plt.legend(leg)
		plt.title(title)
		plt.xlabel("Function Evals")
		plt.ylabel("Fitness Score")
		plt.grid()
		plt.savefig(plt_name + '.png',bbox_inches='tight')
		plt.close()

def test_train(x, y, train_size=0.8):

	# split into train and validation sets
	sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
	for train_index, test_index in sss.split(x, y):
		x_train = x[train_index]
		y_train = y[train_index]
		x_test = x[test_index]
		y_test = y[test_index]

	return x_train, y_train, x_test, y_test

# def kurtosis(x):

# 	n = np.shape(x)[0]
# 	mean = np.sum((x**1)/n) # Calculate the mean
# 	var = np.sum((x-mean)**2)/n # Calculate the variance
# 	skew = np.sum((x-mean)**3)/n # Calculate the skewness
# 	kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
# 	kurt = kurt/(var**2)-3
# 	return kurt, skew, var, mean


def mean_kurtosis(x):

	n_comp = np.shape(x)[1]
	kurts = []
	for i in range(n_comp):
		kurts.append(kurtosis(x[:, i]))
	# print(kurts)
	return np.mean(kurts)

def plot_dimRed(n, metric, ylabel, title, filename):

	plt.plot(n, metric)
	plt.title(title)
	plt.xlabel("number of components")
	plt.ylabel(ylabel)
	plt.xticks(np.array(n))
	plt.grid()
	plt.savefig(filename + '.png',bbox_inches='tight')
	plt.close()

def mean_squared_error(x1, x2):

	return np.sum((x1-x2)**2, axis=1).mean()




