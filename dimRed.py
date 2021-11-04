from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score
from sklearn.model_selection import learning_curve
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from getData import getData
import utils
import sys, getopt
import time
from clusters import run_Kmeans, plot_Kmeans, overplot_Kmeans, run_GMM, plot_GMM, overplot_GMM, plot_clusters


# get data
x_wine, y_wine = getData.wineData()
x_wine = preprocessing.scale(x_wine)
feature_space_wine = np.array(range(1,np.shape(x_wine)[1]+1))

x_cancer, y_cancer = getData.cancerData()
x_cancer = preprocessing.scale(x_cancer)
feature_space_cancer = np.array(range(1,np.shape(x_cancer)[1]+1))

DATASETS = []
ALGS = []
CLUSTERS = []
clusterFlg = False

for i in range(len(sys.argv)):
	if sys.argv[i] == '-a':
		if sys.argv[i+1] == 'PCA':
			ALGS = [0]
		elif sys.argv[i+1] == 'ICA':
			ALGS = [1]
		elif sys.argv[i+1] == 'RP':
			ALGS = [2]
		elif sys.argv[i+1] == 'SVD':
			ALGS = [3]
		# DATASETS = [int(sys.argv[i+1])]
	if sys.argv[i] == '-d':
		if sys.argv[i+1] == 'all':
			DATASETS = [0, 1]
		if sys.argv[i+1] == 'wine':
			DATASETS = [0]
		elif sys.argv[i+1] == 'cancer':
			DATASETS = [1]
		# DATASETS = [int(sys.argv[i+1])]
	if sys.argv[i] == '-c':
		clusterFlg = True
		if sys.argv[i+1] == 'all':
			CLUSTERS = [0, 1]
		elif sys.argv[i+1] == 'kmeans':
			CLUSTERS = [0]
		elif sys.argv[i+1] == 'gmm':
			CLUSTERS = [1]

if len(DATASETS) == 0:
	DATASETS = [0, 1]
if len(ALGS) == 0:
	ALGS = [0, 1, 2, 3]


for alg in ALGS:
	if alg == 0:
		for dat in DATASETS:
			if dat == 0:

				# Get % explained varience

				pca = PCA()
				pca.fit(x_wine)
				x_wine_ = pca.transform(x_wine)
				evr = pca.explained_variance_ratio_ * 100
				evr_cum = np.cumsum(evr)
				feature_space_wine = np.array(range(1,np.shape(x_wine)[1]+1))

				plt.plot(feature_space_wine, evr_cum)
				plt.title('PCA: Wine Data Explained Varience Ratio')
				plt.xlabel("PCA Component")
				plt.ylabel("Explained Varience Ratio (%)")
				plt.xticks(feature_space_wine)
				plt.grid()
				plt.savefig('PCA_WINE_EVR.png',bbox_inches='tight')
				plt.close()


				# pca = PCA(n_components=7)
				# pca.fit(x_wine)
				# x_wine_ = pca.transform(x_wine)

				# idx = 1
				# for eigenvalue in pca.explained_variance_ratio_:
				# 	print("eigenvalue for component #", idx, ": ", eigenvalue)
				# 	# plt.plot(features_len, eigenvector)
				# 	# labels.append("EV #" + str(idx))
				# 	idx = idx + 1
				# print(pca.explained_variance_ratio_)


				# idx = 1
				# labels = []
				# features_len = np.array(range(1,np.shape(x_wine)[1]+1))
				# for eigenvector in pca.components_:
				# 	print("eigenvector for component #", idx, ": ", eigenvector)
				# 	plt.plot(features_len, eigenvector)
				# 	labels.append("EV #" + str(idx))
				# 	idx = idx + 1
				# plt.title('PCA: Wine Data Eigenvectors')
				# plt.xlabel("X feature")
				# plt.legend(labels)
				# # plt.ylabel("Explained Varience Ratio (%)")
				# plt.xticks(features_len)
				# plt.grid()
				# plt.savefig('PCA_WINE_Eigenvectors.png',bbox_inches='tight')
				# plt.close()


				# print(np.shape(np.array(eigenvector)))
				# print(np.shape(pca.explained_variance_ratio_))
				# test = np.matmul(np.array(pca.components_).T, np.array(pca.explained_variance_ratio_))
				# print(test)



				# cRange = range(1, np.shape(x_wine)[1])
				# for c in cRange:

				# 	pca = PCA(n_components=c)
				# 	pca.fit(x_wine)
				# 	x_wine_new = pca.transform(x_wine)



				# print(x_wine)

				# outDF = pd.DataFrame(x_wine_new)
				# oldDF = pd.DataFrame(x_wine)

				# outDF.to_excel('x_wine_new.xlsx')
				# oldDF.to_excel('x_wine.xlsx')

				if clusterFlg:

					n_comp = 7
					pca = PCA(n_components=n_comp)
					pca.fit(x_wine)
					x_wine_ = pca.transform(x_wine)

					plot_clusters(x_wine, x_wine_, y_wine, 'PCA WineData', 'PCA_WineData', CLUSTERS, n_comp)


			if dat == 1:

				pca = PCA()
				pca.fit(x_cancer)
				x_wine_ = pca.transform(x_cancer)
				evr = pca.explained_variance_ratio_ * 100
				evr_cum = np.cumsum(evr)

				plt.plot(feature_space_cancer, evr_cum)
				plt.title('PCA: Cancer Data Explained Varience Ratio')
				plt.xlabel("PCA Component")
				plt.ylabel("Explained Varience Ratio (%)")
				plt.xticks(feature_space_cancer)
				plt.grid()
				plt.savefig('PCA_CANCER_EVR.png',bbox_inches='tight')
				plt.close()


				if clusterFlg:

					n_comp = 6
					pca = PCA(n_components=n_comp)
					# pca = PCA(n_components=2)
					pca.fit(x_cancer)
					x_cancer_ = pca.transform(x_cancer)

					plot_clusters(x_cancer, x_cancer_, y_cancer, 'PCA CancerData', 'PCA_CancerData', CLUSTERS, n_comp)


	if alg == 1:
		for dat in DATASETS:
			if dat == 0:

				# Get % explained varience
				kurt = []
				rec_error = []
				n = []
				for k in range(2,np.shape(x_wine)[1]+1):
					ica = FastICA(n_components=k)
					ica.fit(x_wine)
					x_wine_ = ica.transform(x_wine)
					# evr = ica.explained_variance_ratio_ * 100
					# evr_cum = np.cumsum(evr)
					kurt.append(utils.mean_kurtosis(x_wine_))
					n.append(k)
					# print(kurt)
					components = ica.components_ 
					p_inverse = np.linalg.pinv(components.T)
					x_wine_ = ica.transform(x_wine)
					reconstructed = x_wine_.dot(p_inverse)
					error = utils.mean_squared_error(x_wine, reconstructed)

					# n.append(k)
					rec_error.append(error)

				plt.plot(n, kurt)
				plt.title('ICA: Wine Data - Kurtosis')
				plt.xlabel("number of components")
				plt.ylabel("Kurtosis")
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('ICA_WINE_Kurtosis.png',bbox_inches='tight')
				plt.close()

				plt.plot(n, rec_error)
				plt.title('ICA: Wine Data - Reconstruction Error')
				plt.xlabel("number of components")
				plt.ylabel("Reconstruction Error")
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('ICA_WINE_Reconstruction_Error.png',bbox_inches='tight')
				plt.close()

				# cRange = range(1, np.sha1pe(x_wine)[1])
				# for c in cRange:

				# 	pca = PCA(n_components=c)
				# 	pca.fit(x_wine)
				# 	x_wine_new = pca.transform(x_wine)



				# print(x_wine)

				# outDF = pd.DataFrame(x_wine_new)
				# oldDF = pd.DataFrame(x_wine)

				# outDF.to_excel('x_wine_new.xlsx')
				# oldDF.to_excel('x_wine.xlsx')

				if clusterFlg:

					n_comp = 8
					ica = FastICA(n_components=n_comp)
					ica.fit(x_wine)
					x_wine_ = ica.transform(x_wine)

					plot_clusters(x_wine, x_wine_, y_wine, 'ICA WineData', 'ICA_WineData', CLUSTERS, n_comp)


			if dat == 1:


								# Get % explained varience
				kurt = []
				n = []
				rec_error = []

				for k in range(2,np.shape(x_cancer)[1]+1):

					ica = FastICA(n_components=k)
					ica.fit(x_cancer)
					x_cancer_ = ica.transform(x_cancer)
					kurt.append(utils.mean_kurtosis(x_cancer_))
					n.append(k)
					components = ica.components_ 
					p_inverse = np.linalg.pinv(components.T)
					x_cancer_ = ica.transform(x_cancer)
					reconstructed = x_cancer_.dot(p_inverse)
					error = utils.mean_squared_error(x_cancer, reconstructed)

					# n.append(k)
					rec_error.append(error)

				plt.plot(n, rec_error)
				plt.title('ICA: Cancer Data - Reconstruction Error')
				plt.xlabel("number of components")
				plt.ylabel("Reconstruction Error")
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('ICA_Cancer_Reconstruction_Error.png',bbox_inches='tight')
				plt.close()

				plt.plot(n, kurt)
				plt.title('ICA: Cancer Data - Kurtosis')
				plt.xlabel("number of components")
				plt.ylabel("Kurtosis")
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('ICA_Cancer_Kurtosis.png',bbox_inches='tight')
				plt.close()


				if clusterFlg:

					n_comp = 2
					ica = FastICA(n_components=n_comp)
					ica.fit(x_cancer)
					x_cancer_ = ica.transform(x_cancer)

					plot_clusters(x_cancer, x_cancer_, y_cancer, 'ICA CancerData', 'ICA_CancerData', CLUSTERS, n_comp)


	if alg == 2:
		for dat in DATASETS:
			if dat == 0:

				TRIALS = list(range(10))
				plt_labels = []
				for trial in TRIALS:
					rec_error = []
					n = []
					eu_dist = []

					for k in range(2,np.shape(x_wine)[1]+1):

						grp = GaussianRandomProjection(n_components=k)
						# x_wine_ = grp.fit_transform(x_wine)


						# data has this shape:  row, col = 4898, 11 
						# random_projection = SparseRandomProjection(n_components=5)

						grp.fit(x_wine)
						components =  grp.components_ # shape=(5, 11) 
						p_inverse = np.linalg.pinv(components.T) # shape=(5, 11) 
						# print(p_inverse.shape)

						#now get the transformed data using the projection components
						x_wine_ = grp.transform(x_wine) #shape=(4898, 5) 
						reconstructed = x_wine_.dot(p_inverse)  #shape=(4898, 11) 
						# print(reconstructed.shape)
						# print(x_wine_.shape)
						# assert  x_wine_.shape ==  reconstructed.shape
						error = utils.mean_squared_error(x_wine, reconstructed)

						n.append(k)
						rec_error.append(error)

						# x_wine_ = grp.transform(x_wine)
						# print(np.shape(x_wine))
						# print(np.shape(reconstructed))
						diff_dist = (euclidean_distances(x_wine) - euclidean_distances(reconstructed))**2
						eu_dist.append(diff_dist.mean())
						# print(np.shape(eu_dist[-1]))
					plt.plot(n, eu_dist)
					plt_labels.append('Trial = ' + str(trial))

				# plt.plot(n, rec_error)
				# plt.title('GRP: Wine Data - Reconstruction Error')
				# plt.xlabel("number of components")
				# plt.ylabel("Reconstruction Error")
				# plt.xticks(np.array(n))
				# plt.grid()
				# plt.savefig('GRP_WINE_Reconstruction_Error.png',bbox_inches='tight')
				# plt.close()

				# plt.plot(n, eu_dist)
				plt.title('GRP: Wine Data - Euclidean Distance')
				plt.xlabel("number of components")
				plt.ylabel("Euclidean Distance")
				plt.legend(plt_labels)
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('GRP_WINE_Euclidean_Distance.png',bbox_inches='tight')
				plt.close()


				if clusterFlg:

					n_comp = 9
					grp = GaussianRandomProjection(n_components=n_comp)
					x_wine_ = grp.fit_transform(x_wine)

					plot_clusters(x_wine, x_wine_, y_wine, 'GRP WineData', 'GRP_WineData', CLUSTERS, n_comp)

			if dat == 1:

				rec_error = []
				n = []
				eu_dist = []

				for k in range(2,np.shape(x_cancer)[1]+1):

					grp = GaussianRandomProjection(n_components=k)
					# x_wine_ = grp.fit_transform(x_wine)


					# data has this shape:  row, col = 4898, 11 
					# random_projection = SparseRandomProjection(n_components=5)

					grp.fit(x_cancer)
					components =  grp.components_ # shape=(5, 11) 
					p_inverse = np.linalg.pinv(components.T) # shape=(5, 11) 
					# print(p_inverse.shape)

					#now get the transformed data using the projection components
					x_cancer_ = grp.transform(x_cancer) #shape=(4898, 5) 
					reconstructed = x_cancer_.dot(p_inverse)  #shape=(4898, 11) 
					# print(reconstructed.shape)
					# print(x_wine_.shape)
					# assert  x_wine_.shape ==  reconstructed.shape
					error = utils.mean_squared_error(x_cancer, reconstructed)
					diff_dist = (euclidean_distances(x_cancer) - euclidean_distances(reconstructed))**2
					eu_dist.append(diff_dist.mean())


					n.append(k)
					rec_error.append(error)

				plt.plot(n, rec_error)
				plt.title('GRP: Cancer Data - Reconstruction Error')
				plt.xlabel("number of components")
				plt.ylabel("Reconstruction Error")
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('GRP_Cancer_Reconstruction_Error.png',bbox_inches='tight')
				plt.close()

				plt.plot(n, eu_dist)
				plt.title('GRP: Cancer Data - Euclidean Distance')
				plt.xlabel("number of components")
				plt.ylabel("Euclidean Distance")
				plt.xticks(np.array(n))
				plt.grid()
				plt.savefig('GRP_Cancer_Euclidean_Distance.png',bbox_inches='tight')
				plt.close()


				if clusterFlg:

					n_comp = 6
					grp = GaussianRandomProjection(n_components=n_comp)
					x_wine_ = grp.fit_transform(x_wine)

					plot_clusters(x_cancer, x_cancer_, y_cancer, 'GRP CancerData', 'GRP_CancerRPData', CLUSTERS, n_comp)

	if alg == 3:
		for dat in DATASETS:
			if dat == 0:

				feature_space_wine = feature_space_wine[:-1]
				vt = VarianceThreshold(1)
				vt.fit(x_wine)
				x_wine_ = vt.transform(x_wine)
				print(np.shape(x_wine))
				print(np.shape(x_wine_))
				# # print(x_wine)
				# sdfsd
				# evr = lda.explained_variance_ratio_ * 100
				# evr_cum = np.cumsum(evr)
				# # feature_space_wine = np.array(range(1,np.shape(x_wine)[1]+1))

				# plt.plot(feature_space_wine, evr_cum)
				# plt.title('SVD: Wine Data Explained Varience Ratio')
				# plt.xlabel("SVD Component")
				# plt.ylabel("Explained Varience Ratio (%)")
				# plt.xticks(feature_space_wine)
				# plt.grid()
				# plt.savefig('SVD_WINE_EVR.png',bbox_inches='tight')
				# plt.close()

				if clusterFlg:

					features = np.shape(x_wine_)[1]
					svd = VarianceThreshold(1)
					svd.fit(x_wine)
					x_wine_ = svd.transform(x_wine)

					plot_clusters(x_wine, x_wine_, y_wine, 'VT WineData', 'VT_WineData', CLUSTERS, features)



			if dat == 1:

				feature_space_cancer = feature_space_cancer[:-1]
				vt = VarianceThreshold(1)
				vt.fit(x_cancer)
				x_cancer_ = vt.transform(x_cancer)
				print(np.shape(x_cancer))
				print(np.shape(x_cancer_))
				# evr = lda.explained_variance_ratio_ * 100
				# evr_cum = np.cumsum(evr)

				# plt.plot(feature_space_cancer, evr_cum)
				# plt.title('SVD: Cancer Data Explained Varience Ratio')
				# plt.xlabel("SVD Component")
				# plt.ylabel("Explained Varience Ratio (%)")
				# plt.xticks(feature_space_cancer)
				# plt.grid()
				# plt.savefig('SVD_Cancer_EVR.png',bbox_inches='tight')
				# plt.close()

				if clusterFlg:

					features = np.shape(x_cancer_)[1]
					svd = VarianceThreshold(1)
					svd.fit(x_cancer)
					x_cancer_ = svd.transform(x_cancer)

					plot_clusters(x_cancer, x_cancer_, y_cancer, 'VT CancerData', 'VT_CancerData', CLUSTERS, features)