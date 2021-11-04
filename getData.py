import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class getData():

	def __init__(self):
		pass

	def wineData():

		wine_data = pd.read_csv('winequality-red.csv', sep = ';', header = None)
		wine_df = pd.DataFrame(wine_data)
		wine_data = np.array(wine_data)
		wine_data = np.array(wine_data[1:][:],dtype='float')
		y_wine = np.array(wine_data[:,-1], dtype='int')
		x_wine = wine_data[:, :-1]

		# x = preprocessing.normalize(x_wine)
		x = x_wine
		y = y_wine

		return x, y

	def defaultData():

		default_data = pd.read_csv('default.csv', sep = ',', header = None)
		default_df = pd.DataFrame(default_data)
		default_data = np.array(default_data)
		default_data = np.array(default_data[:][:])
		y_default = default_data[2:, -1]
		x_default = default_data[2:, 1:-1]

		x = np.array(x_default, dtype='int')
		y = np.array(y_default, dtype='int')

		return x, y

	def cancerData():

		cancer_data = pd.read_csv('cancer.csv', sep = ',', header = None)
		cancer_df = pd.DataFrame(cancer_data)
		cancer_data = np.array(cancer_data)
		cancer_data = np.array(cancer_data[:][:])
		# clean data by remvoing rows with incomplete data
		incmp = np.where(cancer_data == '?')
		cancer_data = np.delete(cancer_data, incmp, axis=0)
		y = np.array(cancer_data[:, -1], dtype='int')
		x = np.array(cancer_data[:, 1:-1], dtype='float')
		
		return x, y