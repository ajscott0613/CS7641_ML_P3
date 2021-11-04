from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from getData import getData
import utils
import sys, getopt
import time
from clusters import run_Kmeans, plot_Kmeans, overplot_Kmeans, run_GMM, plot_GMM, overplot_GMM, plot_clusters
from metrics import Metrics
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder			

import warnings
warnings.filterwarnings("ignore")


def runNN(x, y, expName):

	iters = np.linspace(5, 300, 30)

	train_scores_plt = []
	test_scores_plt = []
	for itr in iters:

		skf = StratifiedKFold(n_splits=5)
		model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=int(itr))
		# print(np.unique(y))
		scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
		print("Iter Num: ", itr)
		train_scores = scores['train_score']
		test_scores = scores['test_score']
		train_scores_avg = np.mean(train_scores)
		test_scores_avg = np.mean(test_scores)
		train_scores_plt.append(train_scores_avg)
		test_scores_plt.append(test_scores_avg)

		print("avg train score: ", train_scores_avg)
		print("avg test score: ", test_scores_avg)
		print("----------------------------------------")

	name = "ANN_Wine_" + expName
	title = "ANN (Wine) - " + expName
	Metrics().plot_learning_curve_itr(name, title, iters, train_scores_plt, test_scores_plt)

def runNN_single(x, y, expName):

	x_train, y_train, x_test, y_test = utils.test_train(x, y)
	model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=250)

	print("--------------------------------------")
	print("	Training Model for " + expName + " ...")
	t0 = time.process_time()
	model.fit(x_train, y_train)
	t1 = time.process_time()
	print("	Total Training time:	", np.round(t1 - t0, 4), " seconds")
	
	y_train_pred = model.predict(x_train)
	y_train_accuracy = accuracy_score(y_train, y_train_pred)
	print("	y_train_accuracy: ", y_train_accuracy)

	y_test_pred = model.predict(x_test)
	y_test_accuracy = accuracy_score(y_test, y_test_pred)
	print("	y_test_accuracy: ", y_test_accuracy)

def runNN_xval(x, y, expName):

	skf = StratifiedKFold(n_splits=5)
	model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(200,200), max_iter=300)
	# print(np.unique(y))
	scores  = cross_validate(model, x, y, cv=skf, scoring='accuracy', return_train_score=True)
	train_scores = scores['train_score']
	test_scores = scores['test_score']
	train_scores_avg = np.mean(train_scores)
	test_scores_avg = np.mean(test_scores)

	print("----------------------------------------")
	print("	Training Model for " + expName + " ...")
	print("avg train score: ", train_scores_avg)
	print("avg test score: ", test_scores_avg)
	print("----------------------------------------")
	return train_scores_avg, test_scores_avg

# x_wine, y_wine = getData.wineData()
# x_wine = preprocessing.scale(x_wine)

# runNN(x_wine, y_wine, 'Baseline')


if __name__ == "__main__":

	baseline = False
	single = True
	acFlg = False
	algsFlg = False
	allFlg = False
	ALGS = []
	AC = []

	for i in range(len(sys.argv)):
		if sys.argv[i] == '-genTab':
			allFlg = True
		if sys.argv[i] == '-lcurve':
			single = False
		if sys.argv[i] == '-baseline':
			baseline = True
		if sys.argv[i] == '-a':
			algsFlg = True
			if sys.argv[i+1] == 'all':
				ALGS = [0, 1, 2, 3]
			if sys.argv[i+1] == 'PCA':
				ALGS = [0]
			elif sys.argv[i+1] == 'ICA':
				ALGS = [1]
			elif sys.argv[i+1] == 'RP':
				ALGS = [2]
			elif sys.argv[i+1] == 'VT':
				ALGS = [3]
		if sys.argv[i] == '-ac':
			acFlg = True
			if sys.argv[i+1] == 'all':
				AC = list(range(8))
			if sys.argv[i+1] == 'PCA':
				if sys.argv[i+2] == 'both':
					AC = [0, 1]
				if sys.argv[i+2] == 'kmeans':
					AC = [0]
				elif sys.argv[i+2] == 'gmm':
					AC = [1]
			elif sys.argv[i+1] == 'ICA':
				if sys.argv[i+2] == 'both':
					AC = [2, 3]
				if sys.argv[i+2] == 'kmeans':
					AC = [2]
				elif sys.argv[i+2] == 'gmm':
					AC = [3]
			elif sys.argv[i+1] == 'RP':
				if sys.argv[i+2] == 'both':
					AC = [4, 5]
				if sys.argv[i+2] == 'kmeans':
					AC = [4]
				elif sys.argv[i+2] == 'gmm':
					AC = [5]
			elif sys.argv[i+1] == 'VT':
				if sys.argv[i+2] == 'both':
					AC = [6, 7]
				if sys.argv[i+2] == 'kmeans':
					AC = [6]
				elif sys.argv[i+2] == 'gmm':
					AC = [7]


	if allFlg:
		baseline = True
		single = True
		acFlg = True
		algsFlg = True
		ALGS = [0, 1, 2, 3]
		AC = list(range(8))
	data = np.zeros((13, 2))


	# get wine data
	x_wine, y_wine = getData.wineData()
	x_wine = preprocessing.scale(x_wine)

	if baseline:
		if single:
			# runNN_single(x_wine, y_wine, 'Baseline')
			out = runNN_xval(x_wine, y_wine, 'Baseline')
			data[0, 0] = np.round(out[0],4)
			data[0, 1] = np.round(out[1],4)

		else:
			runNN(x_wine, y_wine, 'Baseline')

	if algsFlg:
		for alg in ALGS:
			if alg == 0:
				comps = 8
				pca = PCA(n_components=comps)
				x_wine_ = pca.fit_transform(x_wine)

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'PCA_C=' + str(comps))
					data[alg+1, 0] = np.round(out[0],4)
					data[alg+1, 1] = np.round(out[1],4)


				else:
					runNN(x_wine_, y_wine, 'PCA_C=' + str(comps))

			elif alg == 1:
				comps = 8
				ica = FastICA(n_components=comps)
				x_wine_ = ica.fit_transform(x_wine)

				if single:
					# runNN_single(x_wine_, y_wine, 'ICA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'ICA_C=' + str(comps))
					data[alg+1, 0] = np.round(out[0],4)
					data[alg+1, 1] = np.round(out[1],4)


				else:
					runNN(x_wine_, y_wine, 'ICA_C=' + str(comps))


			elif alg == 2:
				comps = 8
				grp = GaussianRandomProjection(n_components=comps)
				x_wine_ = grp.fit_transform(x_wine)

				if single:
					# runNN_single(x_wine_, y_wine, 'GRP_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'GRP_C=' + str(comps))
					data[alg+1, 0] = np.round(out[0],4)
					data[alg+1, 1] = np.round(out[1],4)


				else:
					runNN_single(x_wine_, y_wine, 'GRP_C=' + str(comps))

			elif alg == 3:
				comps = 8
				vt = VarianceThreshold(1)
				x_wine_ = vt.fit_transform(x_wine)

				if single:
					# runNN_single(x_wine_, y_wine, 'GRP_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'VT_C=' + str(comps))
					data[alg+1, 0] = np.round(out[0],4)
					data[alg+1, 1] = np.round(out[1],4)


				else:
					runNN_single(x_wine_, y_wine, 'VT_C=' + str(comps))

	if acFlg:
		for ac in AC:
			if ac == 0:
				comps = 8
				k = 5
				pca = PCA(n_components=comps)
				x_wine_ = pca.fit_transform(x_wine)
				kmeans = KMeans(n_clusters=k, random_state=5, n_init=1, max_iter=1000)
				kmeans.fit(x_wine_)
				print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(kmeans.labels_.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'PCA_C=' + str(comps) + ', Kmeans_k=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)


				else:
					runNN(x_wine_, y_wine, 'PCA_C=' + str(comps))

			if ac == 1:
				comps = 8
				k = 5
				pca = PCA(n_components=comps)
				x_wine_ = pca.fit_transform(x_wine)
				gmm = GaussianMixture(n_components=k, random_state=5, n_init=1, max_iter=1000)
				gmm.fit(x_wine_)
				labels = gmm.predict(x_wine_)
				# print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(labels.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				# print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				# print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				# print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'PCA_C=' + str(comps) + ', GMM_C=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)


				else:
					runNN(x_wine_, y_wine, 'PCA_C=' + str(comps))

			if ac == 2:
				comps = 8
				k = 5
				ica = FastICA(n_components=comps)
				x_wine_ = ica.fit_transform(x_wine)
				kmeans = KMeans(n_clusters=k, random_state=5, n_init=1, max_iter=1000)
				kmeans.fit(x_wine_)
				print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(kmeans.labels_.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'ICA_C=' + str(comps) + ', Kmeans_k=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)


				else:
					runNN(x_wine_, y_wine, 'ICA_C=' + str(comps))

			if ac == 3:
				comps = 8
				k = 5
				ica = FastICA(n_components=comps)
				x_wine_ = ica.fit_transform(x_wine)
				gmm = GaussianMixture(n_components=k, random_state=5, n_init=1, max_iter=1000)
				gmm.fit(x_wine_)
				labels = gmm.predict(x_wine_)
				# print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(labels.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				# print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				# print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				# print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'ICA_C=' + str(comps) + ', GMM_C=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)


				else:
					runNN(x_wine_, y_wine, 'ICA_C=' + str(comps))

			if ac == 4:
				comps = 9
				k = 5
				ica = GaussianRandomProjection(n_components=comps)
				x_wine_ = ica.fit_transform(x_wine)
				kmeans = KMeans(n_clusters=k, random_state=5, n_init=1, max_iter=1000)
				kmeans.fit(x_wine_)
				print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(kmeans.labels_.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'GRP_C=' + str(comps) + ', Kmeans_k=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)

	
				else:
					runNN(x_wine_, y_wine, 'GRP_C=' + str(comps))

			if ac == 5:
				comps = 4
				k = 5
				ica = GaussianRandomProjection(n_components=comps)
				x_wine_ = ica.fit_transform(x_wine)
				gmm = GaussianMixture(n_components=k, random_state=5, n_init=1, max_iter=1000)
				gmm.fit(x_wine_)
				labels = gmm.predict(x_wine_)
				# print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(labels.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				# print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				# print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				# print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'GRP_C=' + str(comps) + ', GMM_C=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)

	
				else:
					runNN(x_wine_, y_wine, 'GRP_C=' + str(comps))


			if ac == 6:
				comps = 4
				k = 5
				ica = VarianceThreshold(1)
				x_wine_ = ica.fit_transform(x_wine)
				kmeans = KMeans(n_clusters=k, random_state=5, n_init=1, max_iter=1000)
				kmeans.fit(x_wine_)
				print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(kmeans.labels_.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'VT_F=' + str(comps) + ', Kmeans_k=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)

	
				else:
					runNN(x_wine_, y_wine, 'GRP_C=' + str(comps))

			if ac == 7:
				comps = 6
				k = 5
				ica = VarianceThreshold(1)
				x_wine_ = ica.fit_transform(x_wine)
				gmm = GaussianMixture(n_components=k, random_state=5, n_init=1, max_iter=1000)
				gmm.fit(x_wine_)
				labels = gmm.predict(x_wine_)
				# print(np.unique(kmeans.labels_))
				one_hot = OneHotEncoder()
				x_newFeatures = one_hot.fit_transform(labels.reshape(-1, 1)).todense()
				# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
				# print(np.shape(x_newFeatures))
				# print(x_newFeatures)
				# print(np.shape(x_wine_))
				x_wine_ = np.hstack((x_wine_, x_newFeatures))
				# print(np.shape(x_wine_))

				if single:
					# runNN_single(x_wine_, y_wine, 'PCA_C=' + str(comps))
					out = runNN_xval(x_wine_, y_wine, 'VT_F=' + str(comps) + ', GMM_C=' + str(k))
					data[ac+5, 0] = np.round(out[0],4)
					data[ac+5, 1] = np.round(out[1],4)

	
				else:
					runNN(x_wine_, y_wine, 'VT_F=' + str(comps))


	print(data)
	# data[0, 0] = 0.9876
	# data[4, 0] = 0.6543
	# allFlg=True
	if allFlg:

		rows = ('Baseline', 'PCA', 'ICA', 'RP', 'VT', 'PCA + kmeans',
			'PCA + GMM', 'ICA + K-means', 'ICA + GMM', 'RP + K-means', 'RP + GMM',
			'VT + K-means', 'VT + GMM')
		columns = ('Average Train Score', 'Average Test Score')
		cell_text = []
		n_rows = len(data)


		# Initialize the vertical-offset for the stacked bar chart.
		y_offset = np.zeros(len(columns))
		for row in range(n_rows):
		    y_offset = y_offset + data[row]
		    cell_text.append([x for x in data[row]])
		# Reverse colors and text labels to display the last value at the top.
		# cell_text.reverse()
		# Add a table at the bottom of the axes
		the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='center')

		ax = plt.gca()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.box(on=None)

		fig = plt.gcf()
		plt.savefig('NN_Testing_Summary.png',
            bbox_inches='tight',
            dpi=150
            )

		df = pd.DataFrame(data)
		df.to_excel('NN_Testing_Summary.xlsx')
		# the_table = plt.table(cellText=cell_text,
  #                     rowLabels=row_headers,
  #                     rowColours=rcolors,
  #                     colLabels=column_headers,
  #                     loc='center')
