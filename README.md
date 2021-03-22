# latent-svm

## Description
This is a numpy-based python implementation of the idea Polson & Scott that reformulates the traditional binary linear SVM problem into a MAP (Maximum a Posteriori) estimation in a probabilistic generative model, and by use of the technique of data augmentation.

You may find the paper here:
"Data Augmentation for Support Vector Machines" by Nicholas G. Polson and Steven L. Scotty, published in Bayesian Analysis (2011)
https://projecteuclid.org/journals/bayesian-analysis/volume-6/issue-1/Data-augmentation-for-support-vector-machines/10.1214/11-BA601.full

Specifically, I implemented the basic EM-SVM algorithm (both stable and unstable versions), the ECME-SVM algorithm and the two variants (with or without a step of learning $\nu$) of the MCMC-SVM algorithm.

The code uses wherever possible numpy but is not further optimized with numba or cython.

## Code layout
The computation methods are included in the following files: unstable_em_svm.py, stable_em_svm.py mcmc_svm.py, ecme_svm.py. The methods are commented in the style of numpy with strict conditions on the dimension shapes.

## Demo
A demo jupyter notebook is included for visualization and testing. It uses two public datasets:
* The Spam dataset, available here https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/. We used the spambase.data file.
* The Credit default dataset, available here https://archive.ics.uci.edu/ml/machine-learning-databases/00350/. We used the default of credit card clients.xls file.
All files must be put in the root of the repository for correct execution.

## Dependencies
The code was tested on an Ubuntu 20 machine with a conda distribution. The algorithm code and demo code worked with these versions:
* python 3.7.7
* numpy 1.18.1
* pandas 1.0.3
* matplotlib 3.3.4
* sklearn 0.24.1

