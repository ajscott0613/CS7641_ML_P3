from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from getData import getData
import utils
import sys, getopt
import time
from clusters import run_Kmeans, plot_Kmeans, overplot_Kmeans, run_GMM, plot_GMM, overplot_GMM, plot_clusters, p1_Keamns, print_mertrics



if __name__ == "__main__":

	baseline = False
	single = False
	printFlg = False
	ALGS = []
	CLUSTERS = []

	for i in range(len(sys.argv)):
		if sys.argv[i] == '-single':
			single = True
		if sys.argv[i] == '-baseline':
			baseline = True
		if sys.argv[i] == '-c':
			if sys.argv[i+1] == 'all':
				CLUSTERS = [0, 1, 2]
			if sys.argv[i+1] == 'kmeans':
				CLUSTERS = [0]
			elif sys.argv[i+1] == 'gmm':
				CLUSTERS = [1]
		if sys.argv[i] == '-d':
			if sys.argv[i+1] == 'all':
				DATASETS = [0, 1]
			if sys.argv[i+1] == 'wine':
				DATASETS = [0]
			elif sys.argv[i+1] == 'cancer':
				DATASETS = [1]
		if sys.argv[i] == '-print_metrics':
			printFlg = True




	# get data
	x_wine, y_wine = getData.wineData()
	x_wine = preprocessing.scale(x_wine)
	feature_space_wine = np.array(range(2,np.shape(x_wine)[1]+1))

	x_cancer, y_cancer = getData.cancerData()
	x_cancer = preprocessing.scale(x_cancer)
	feature_space_cancer = np.array(range(2,np.shape(x_cancer)[1]+1))

	for clust in CLUSTERS:
		if clust == 0:
			for dat in DATASETS:
				if dat == 0:
					p1_Keamns(x_wine, y_wine, 'KMeans Wine Analysis', 'KMeans_Wine_Analysis', 
						kRange=feature_space_wine, verbose=False)
					print_mertrics(x_wine, y_wine, [7], alg='Kmeans')

				if dat == 1:
					p1_Keamns(x_cancer, y_cancer, 'KMeans Cancer Analysis', 'KMeans_Cancer_Analysis', 
						kRange=feature_space_cancer, verbose=False)
		if clust == 1:
			for dat in DATASETS:
				if dat == 0:
					gmmOut = run_GMM(x_wine, y_wine, kRange=feature_space_wine, verbose=True)
					gmm, n, aic_scores, bic_scores = gmmOut
					plot_GMM(n, aic_scores, bic_scores, 'GMM Wine Analysis', 'GMMs_Wine_Analysis')

				if dat == 1:
					gmmOut = run_GMM(x_cancer, y_cancer, kRange=feature_space_cancer, verbose=True)
					gmm, n, aic_scores, bic_scores = gmmOut
					plot_GMM(n, aic_scores, bic_scores, 'GMM Cancer Analysis', 'GMMs_Cancer_Analysis')


	if printFlg:
		# print_mertrics(x_wine, y_wine, [feature_space_wine[-1]], alg='Kmeans')
		print_mertrics(x_wine, y_wine, [7], alg='Kmeans')
		# print_mertrics(x_cancer, y_cancer, [feature_space_cancer[-1]], alg='Kmeans')
		print_mertrics(x_cancer, y_cancer, [6], alg='Kmeans')

		# print_mertrics(x_wine, y_wine, [feature_space_wine[-1]], alg='GMM')
		print_mertrics(x_wine, y_wine, [7], alg='GMM')
		# print_mertrics(x_cancer, y_cancer, [feature_space_cancer[-1]], alg='GMM')
		print_mertrics(x_cancer, y_cancer, [7], alg='GMM')