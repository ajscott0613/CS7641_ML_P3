from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Metrics():

	def __init__(self, k_folds=5, train_sizes=np.linspace(0.01, 1, 100), verbose=False):

		self.k_folds = k_folds
		self.train_sizes = train_sizes
		self.verbose = verbose

	def learning_curve_data(self, model, x, y):

		skf = StratifiedKFold(n_splits=self.k_folds)
		lcurve = learning_curve(model, x, y, cv=skf, scoring='accuracy', train_sizes=self.train_sizes, verbose=self.verbose)
		train_sizes_tot, train_scores, test_scores = lcurve
		self.train_scores_mean = np.mean(train_scores, axis=1)
		self.test_scores_mean = np.mean(test_scores, axis=1)
		self.train_sizes_tot = train_sizes_tot

	def plot_learning_curve(self, name, title):

		plt_name = name
		plt.plot(self.train_sizes_tot, self.train_scores_mean)
		plt.plot(self.train_sizes_tot, self.test_scores_mean)
		plt.legend(["train_data", "validation_data"])
		plt.title(title)
		plt.xlabel("Training Data Size")
		plt.ylabel("Accuracy")
		plt.grid()
		plt.savefig(plt_name + '.png',bbox_inches='tight')
		plt.close()

	def dsiplay_score(self, model, x_trn, y_trn, x_tst, y_tst):

		model.fit(x_tst, y_tst)
		trn_score = model.score(x_tst, y_tst)
		tst_score = model.score(x_trn, y_trn)
		print("----------------------------------")
		print("Training Score: ", trn_score)
		print("Testing Score: ", tst_score)
		print("----------------------------------")


	def cross_validation_scores(self, model, param_grid, x, y, name):

		clf = GridSearchCV(model, param_grid, cv=self.k_folds)
		clf.fit(x,y)
		df = pd.DataFrame(clf.cv_results_)
		print(df)
		df.to_excel(name + '.xlsx')

	def plot_learning_curve_itr(self, name, title, itrs, trn, tst):

		plt_name = name
		plt.plot(itrs, trn)
		plt.plot(itrs, tst)
		plt.legend(["train_data", "validation_data"])
		plt.title(title)
		plt.xlabel("Number of Iterations")
		plt.ylabel("Accuracy")
		plt.grid()
		plt.savefig(plt_name + '.png',bbox_inches='tight')
		plt.close()



