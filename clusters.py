from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score, v_measure_score
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


def print_mertrics(x, y, kRange, alg='Kmeans'):

	for k in kRange:
		if alg == 'Kmeans':
			cluster = KMeans(n_clusters=k, random_state=5, n_init=1, max_iter=1000)
		elif alg == 'GMM':
			cluster = GaussianMixture(n_components=k, random_state=5, n_init=1, max_iter=1000, init_params='kmeans')
		cluster.fit(x)
		label = cluster.predict(x)
		sil_score = silhouette_score(x, label)
		# comp_score = completeness_score(y, cluster.labels_)
		# homg_score = homogeneity_score(y, cluster.labels_)
		# vmeas_score = v_measure_score(y, cluster.labels_)
		comp_score = completeness_score(y, label)
		homg_score = homogeneity_score(y, label)
		vmeas_score = v_measure_score(y, label)

		print("Running for " + alg)
		print("K Value = ", k, ":	sil_score = ", sil_score, ", comp_score = ", 
			comp_score, ", homg_score = ", homg_score, ", vmeas_score = ", vmeas_score)
		print("-------------------------------------------------------")


def p1_Keamns(x, y, title, fname, kRange=range(2,10), verbose=False):


	sil_scores = []
	wcss = []
	n = []

	# kRange = range(2,4)
	for k in kRange:

		kmeans = KMeans(n_clusters=k, random_state=5, n_init=1000, max_iter=1000)
		kmeans.fit(x)
		label = kmeans.predict(x)
		sil_scores.append(silhouette_score(x, label))
		wcss.append(kmeans.inertia_)
		n.append(k)

	plt.plot(n, sil_scores)
	plt.title(title)
	plt.ylabel("Average Silhouette Score")
	plt.xlabel("Number of Clusters")
	plt.xticks(np.array(n))
	# plt.ylabel("Accuracy")
	plt.grid()
	plt.savefig(fname + '.png',bbox_inches='tight')
	plt.close()


	fig, axs = plt.subplots(2, 1)
	print(axs)
	print(n)
	print(sil_scores)
	axs[0].plot(n, sil_scores)
	axs[0].set(ylabel='silhouette score')
	axs[0].set(xticks=np.array(n))
	# axs[0].set(xlabel='number of clusters')
	axs[0].grid()
	axs[1].plot(n, wcss)
	axs[1].set(ylabel='Within-Cluster Sum of Square')
	axs[1].set(xticks=np.array(n))
	axs[1].set(xlabel='number of clusters')
	axs[1].grid()

	fig.suptitle(title)
	fig.savefig(fname + '_both.png',bbox_inches='tight')
	plt.close()


	# if verbose:
	# 	print("Progress: ", int((k / kRange[-1]) * 100), "% -->	k = ", k, ", sil_score = ", sil_scores[-1], ", comp_score = ", 
	# 		comp_scores[-1], ", homg_score = ", homg_scores[-1], ", vmeas_score = ", vmeas_scores[-1])


def run_Kmeans(x, y, kRange=range(2,10), verbose=False):

			# kmeans = KMeans(n_clusters=30, random_state=5, n_init=1000, max_iter=1000)
		sil_scores = []
		comp_scores = []
		homg_scores = []
		vmeas_scores = []
		n = []

		for k in kRange:

			kmeans = KMeans(n_clusters=k, random_state=5, n_init=1, max_iter=1000)
			kmeans.fit(x)
			label = kmeans.predict(x)
			sil_scores.append(silhouette_score(x, label))
			comp_scores.append(completeness_score(y, kmeans.labels_))
			homg_scores.append(homogeneity_score(y, kmeans.labels_))
			vmeas_scores.append(v_measure_score(y, kmeans.labels_))
			n.append(k)

			if verbose:
				print("Progress: ", int((k / kRange[-1]) * 100), "% -->	k = ", k, ", sil_score = ", sil_scores[-1], ", comp_score = ", 
					comp_scores[-1], ", homg_score = ", homg_scores[-1], ", vmeas_score = ", vmeas_scores[-1])
				print("-------------------------------------------------------")

		return kmeans, n, sil_scores, comp_scores, homg_scores, vmeas_scores


def plot_Kmeans(n, sil_scores, comp_scores, homg_scores, vmeas_scores, title, fname):

	fig, axs = plt.subplots(2, 2)
	axs[0, 0].plot(n, sil_scores)
	axs[0, 0].set_title('silhouette score')
	axs[0, 0].grid()

	axs[0, 1].plot(n, comp_scores, 'tab:orange')
	axs[0, 1].set_title('completeness score')
	axs[0, 1].grid()

	axs[1, 0].plot(n, homg_scores, 'tab:green')
	axs[1, 0].set_title('homogeneity score')
	axs[1, 0].set(xlabel='number of clusters')
	axs[1, 0].set(xticks=np.array(n))
	axs[1, 0].grid()

	axs[1, 1].plot(n, vmeas_scores, 'tab:purple')
	axs[1, 1].set_title('v-measure score')
	axs[1, 1].set(xlabel='number of clusters')
	axs[1, 1].set(xticks=np.array(n))
	axs[1, 1].grid()
	# axs[1, 1].plot(x, -y, 'tab:red')
	# axs[1, 1].set_title('Axis [1, 1]')

	# for ax in axs.flat:
	# 	ax.set(xlabel='number of clusters')
	plt.setp(axs[0, 0].get_xticklabels(), visible=False)
	plt.setp(axs[0, 1].get_xticklabels(), visible=False)
	# # plt.setp(axs[0, 0].get_xlabel(), visible=False)
	# # plt.setp(axs[0, 1].get_xlabel(), visible=False)

	# # Hide x labels and tick labels for top plots and y ticks for right plots.
	# print(axs.flat)
	# for ax in axs.flat:
	# 	print(ax)
	# 	ax.label_outer()

	# plt.setp(axs[0, 0].get_yticklabels(), visible=True)
	# plt.setp(axs[0, 1].get_yticklabels(), visible=True)


	fig.suptitle(title)
	# fig.savefig(fname + '.png',bbox_inches='tight')
	fig.savefig(fname + '.png',bbox_inches=0)
	plt.close()


def overplot_Kmeans(n, sil_scores, comp_scores, homg_scores, vmeas_scores, title, fname, n_comp):

	fig, axs = plt.subplots(2, 2)
	axs[0, 0].plot(n[0], sil_scores[0])
	axs[0, 0].plot(n[1], sil_scores[1])
	# axs[0, 0].legend('base', 'PCA')
	axs[0, 0].set_title('silhouette score')
	axs[0, 0].set(xticks=np.array(n[1]))
	axs[0, 0].grid()
	axs[0, 1].plot(n[0], comp_scores[0])
	axs[0, 1].plot(n[1], comp_scores[1])
	# axs[0, 1].legend('base', 'PCA')
	axs[0, 1].set_title('completeness score')
	axs[0, 1].set(xticks=np.array(n[1]))
	axs[0, 1].grid()
	axs[1, 0].plot(n[0], homg_scores[0])
	axs[1, 0].plot(n[1], homg_scores[1])
	# axs[1, 0].legend('base', 'PCA')
	axs[1, 0].set_title('homogeneity score')
	axs[1, 0].set(xlabel='number of clusters')
	axs[1, 0].set(xticks=np.array(n[1]))
	axs[1, 0].grid()

	axs[1, 1].plot(n[0], vmeas_scores[0])
	axs[1, 1].plot(n[1], vmeas_scores[1])
	# axs[1, 0].legend('base', 'PCA')
	axs[1, 1].set_title('v-measure score')
	axs[1, 1].set(xlabel='number of clusters')
	axs[1, 1].set(xticks=np.array(n[1]))
	axs[1, 1].grid()
	# axs[1, 1].plot(x, -y, 'tab:red')
	# axs[1, 1].set_title('Axis [1, 1]')

	# plt.xlabel('number of clusters')

	# for ax in axs.flat:
	# 	ax.set(xlabel='number of clusters')
	plt.setp(axs[0, 0].get_xticklabels(), visible=False)
	plt.setp(axs[0, 1].get_xticklabels(), visible=False)
	# plt.setp(axs[0, 0].get_xlabel(), visible=False)
	# plt.setp(axs[0, 1].get_xlabel(), visible=False)

	labels = ['reduced (' + str(n_comp) + ' components)', 'base']
	fig.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.1), ncol=len(labels), bbox_transform=fig.transFigure)

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	# for ax in axs.flat:
	# 	ax.label_outer()


	fig.suptitle(title)
	fig.savefig(fname + '.png',bbox_inches='tight')
	plt.close()


def run_GMM(x, y, kRange=range(2,10), verbose=False):


	scores = []
	aic_scores = []
	bic_scores = []
	n = []

	for k in kRange:

		gmm = GaussianMixture(n_components=k)
		gmm.fit(x)
		# label = gmm.predict(x)
		# scores.append(silhouette_score(x, label))
		aic_scores.append(gmm.aic(x))
		bic_scores.append(gmm.bic(x))
		n.append(k)


		if verbose:
			print("Progress: ", int((k / kRange[-1]) * 100), "% -->	k = ", k, ", aic_score = ", aic_scores[-1], ", bic_score = ", 
				bic_scores[-1])
			print("-------------------------------------------------------")

	return gmm, n, aic_scores, bic_scores


def run_GMM_metrics(x, y, kRange=range(2,10), verbose=False):


	sil_scores = []
	comp_scores = []
	homg_scores = []
	vmeas_scores = []
	n = []

	for k in kRange:

		gmm = GaussianMixture(n_components=k, random_state=5, n_init=1, max_iter=1000)
		gmm.fit(x)
		label = gmm.predict(x)
		sil_scores.append(silhouette_score(x, label))
		comp_scores.append(completeness_score(y, label))
		homg_scores.append(homogeneity_score(y, label))
		vmeas_scores.append(v_measure_score(y, label))
		n.append(k)

		if verbose:
			print("Progress: ", int((k / kRange[-1]) * 100), "% -->	k = ", k, ", sil_score = ", sil_scores[-1], ", comp_score = ", 
				comp_scores[-1], ", homg_score = ", homg_scores[-1], ", vmeas_score = ", vmeas_scores[-1])
			print("-------------------------------------------------------")

	return gmm, n, sil_scores, comp_scores, homg_scores, vmeas_scores


def plot_GMM(n, aic_scores, bic_scores, title, fname):


	# plt_name = name
	plt.plot(n, aic_scores)
	plt.plot(n, bic_scores)
	plt.legend(["aic", "bic"])
	plt.title(title)
	plt.xlabel("number of components")
	plt.xticks(np.array(n))
	# plt.ylabel("Accuracy")
	plt.grid()
	plt.savefig(fname + '.png',bbox_inches='tight')
	plt.close()

def overplot_GMM(n, aic_scores, bic_scores, title, fname, n_comp):


	# plt_name = name
	plt.plot(n[0], aic_scores[0])
	plt.plot(n[1], aic_scores[1])
	plt.plot(n[0], bic_scores[0])
	plt.plot(n[1], bic_scores[1])
	plt.legend(["aic: reduced (" + str(n_comp) + ")", "aic: base", "bic: reduced (" + str(n_comp) + ")", "bic: base",])
	plt.title(title)
	plt.xlabel("number of components")
	plt.xticks(np.array(n[1]))
	# plt.ylabel("Accuracy")
	plt.grid()
	plt.savefig(fname + '.png',bbox_inches='tight')
	plt.close()


def plot_clusters(x, x_, y, title, filename, clusters, n_comp):

	fname = filename
	t = title

	for clust in clusters:

		if clust == 0:

			# kRange = range(2, np.shape(x_)[1])
			kRange = range(2, np.shape(x)[1]+1)
			kOut = run_Kmeans(x_, y, kRange=kRange, verbose=True)
			kmeans, n, sil_scores, comp_scores, homg_scores, vmeas_scores = kOut
			filename = fname + '_Kmeans_Reduced'
			title = t + ' KMeans'
			plot_Kmeans(n, sil_scores, comp_scores, homg_scores, vmeas_scores, title, filename)

			kRange = range(2, np.shape(x)[1]+1)
			kOut_base = run_Kmeans(x, y, kRange=kRange, verbose=True)
			kmeans_base, n_base, sil_scores_base, comp_scores_base, homg_scores_base, vmeas_scores_base = kOut_base
			filename = fname + '_Kmeans_Overplot'
			title = t + ' KMeans'
			overplot_Kmeans((n, n_base), (sil_scores, sil_scores_base), 
				(comp_scores, comp_scores_base), (homg_scores, homg_scores_base), 
				(vmeas_scores, vmeas_scores_base), title, filename, n_comp)

		if clust == 1:

			# # kRange = range(2, np.shape(x_)[1]+1)
			# kRange = range(2, np.shape(x)[1]+1)
			# gmmOut = run_GMM(x_, y, kRange=kRange, verbose=True)
			# gmm, n, aic_scores, bic_scores = gmmOut
			# filename = fname + '_GMM_Reduced'
			# title = t + ' GMM'
			# plot_GMM(n, aic_scores, bic_scores, title, filename)

			# kRange = range(2, np.shape(x)[1]+1)
			# gmmOut_base = run_GMM(x, y, kRange=kRange, verbose=True)
			# gmm_base, n_base, aic_scores_base, bic_scores_base = gmmOut_base
			# filename = fname + '_GMM_Overplot'
			# title = t + ' GMM'
			# overplot_GMM((n, n_base), (aic_scores, aic_scores_base), 
			# 	(bic_scores, bic_scores_base), title, filename, n_comp)

			kRange = range(2, np.shape(x)[1]+1)
			gmmOut = run_GMM_metrics(x_, y, kRange=kRange, verbose=True)
			gmm, n, sil_scores, comp_scores, homg_scores, vmeas_scores = gmmOut
			filename = fname + '_GMM_Reduced'
			title = t + ' GMM'
			plot_Kmeans(n, sil_scores, comp_scores, homg_scores, vmeas_scores, title, filename)

			kRange = range(2, np.shape(x)[1]+1)
			gmmOut_base = run_GMM_metrics(x, y, kRange=kRange, verbose=True)
			gmm_base, n_base, sil_scores_base, comp_scores_base, homg_scores_base, vmeas_scores_base = gmmOut_base
			filename = fname + '_GMM_Overplot'
			title = t + ' GMM'
			overplot_Kmeans((n, n_base), (sil_scores, sil_scores_base), 
				(comp_scores, comp_scores_base), (homg_scores, homg_scores_base), 
				(vmeas_scores, vmeas_scores_base), title, filename, n_comp)
