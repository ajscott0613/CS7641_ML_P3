# CS7641_ML_P3
Machine Learning Project 3

This repo can be cloned by running "git clone https://github.com/ajscott0613/CS7641_ML_P3.git"

The requirements.txt file can be installed via the command line by running "pip install -r requirments" and contains the necessary packages and version to run the python files.

Each of the files noted in the instructions section can be ran via the command line and requirement specific command line inputs to run them.

## File Contents

runClusters.py: Thie file contains the code to generate summary data and plots for differnet clustering algorithms.

dimRed.py: This file contians the code to generate summary data and plots for the dimension reduction algorithms.

NN.py:  This file contains the code to generate either summary data or learning curves for neural network performance with differnet types of processed data.

clusters.py:  Contians helper function for evaluating and plotting K-Means and GMM cluster algorithms.

utils.py: Contains support functions for plotting.

metrics.py:  Contains support function for generating data and plotting.

getData.py: Used to extract and preprocess data used for the experiments.

winequality-red.csv: Wine data set used for experiments.

cancer.csv: Cancer data set used for experiments.

requirements.txt: contains the packages needed to run the files

## Instructions

runClusters.py can be ran from the command line by running <python3 runClusters.py "args">
The following arguments can be taken:
-c all, -c kmeans, -c gmm
-d all, -d wine, -d cancer
-printmetrics

-c specificies the cluster or all can be ran
-d specificies the dataset of ball can be ran
-prinmetrics prints clusters metrics for a single k cluster value specified in the code.


dimRed.py can be ran from the command line by running <python3 dimRed.py "args">
The following arguments can be taken:
-a all, -a PCA, -a ICA, -a RP, -a VT
-c all, -c kmeans, -c gmm
-d all, -d wine, -d cancer

-c specificies the cluster or all can be ran
-d specificies the dataset of all can be ran
-a specifies the dimensionality reduction algorithm or all can be ran


NN.py can be ran from the command line by running <python3 NN.py "args">
The following arguments can be taken:
-a all, -a PCA, -a ICA, -a RP, -a VT
-ac, can be ran as -ac all, or specific combinations can be selected such as -ac PCA both, or -ac ICA kmeans. 
-lcurve
-genTab

-a specifies the dimensionality reduction algorithm or all can be ran and prints the output.
-ac specifies a dimensionality reduction and clustering combination that can be ran and prints the output.
-genTab runs all combinations of data and generates a table
-lcurve genereates a learning curve and can be ran with -a or -ac
