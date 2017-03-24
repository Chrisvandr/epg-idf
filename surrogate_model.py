import numpy as np
import getpass
import time
import pandas as pd

#import scipy
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Sensitivity modulespip install
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

#Sci-kit modules
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge # Linear Regression / Ordinary Least Squares, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression # Partial Least Squares Regression
from sklearn.svm import SVR, LinearSVR #Support Vector Regression
from sklearn.kernel_ridge import KernelRidge # Kernel Ridge Regression
from sklearn.gaussian_process import GaussianProcessRegressor # Kriging/GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C # GPR kernels
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

UserName = getpass.getuser()
DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'
df = pd.read_csv(DataPath+ 'test_data.csv', header=0)
cols = df.columns.tolist()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b','#d62728','#9467bd','#aec7e8','#ffbb78','#98df8a','#c49c94','#ff9896','#c5b0d5','#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b','#d62728','#9467bd','#aec7e8','#ffbb78','#98df8a','#c49c94','#ff9896','#c5b0d5']

X_real = df.ix[:,0:20]
X_real = X_real.as_matrix()
Y_real = df.ix[:,21:23]
Y_real = Y_real.as_matrix()
print(X_real.shape, Y_real.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real) #randomly split your data
print(X_train.shape, Y_train.shape)
print(Y_train.ravel().shape)


def correlations():
    fig = plt.figure(0)
    ax = plt.subplot(111)

    print(cols[0], cols[21])
    print(spearmanr(X_real[:,0], Y_real[:,0])[0])
    print(pearsonr(X_real[:, 0], Y_real[:, 0])[0])
    bar_width = .4
    spearman_list, pearsonr_list = [], []
    y_pos = np.arange(X_real.shape[1])
    for i in xrange(X_real.shape[1]):
        spearman_list.append(spearmanr(X_real[:, i], Y_real[:, 0])[0])
        pearsonr_list.append(pearsonr(X_real[:, i], Y_real[:, 0])[0])

    ax.barh(y_pos, spearman_list, bar_width,
            color='#1f77b4',
            label='Spearman')
    ax.barh(y_pos+bar_width, pearsonr_list, bar_width,
            color='#ff7f0e',
            label='Pearson')

    plt.xlabel('Correlation')
    plt.yticks(y_pos + bar_width / 2, cols)
    plt.legend()
    #plt.tight_layout()
    plt.show()
#correlations()


#Sobol does not work for the test data as it is sampled by LHS and it needs to be sampled by saltelli sampling (which creates many more variables).
def sobol_analysis():
    # https://media.readthedocs.org/pdf/salib/latest/salib.pdf | http://salib.github.io/SALib/
    # https://waterprogramming.wordpress.com/2013/08/05/running-sobol-sensitivity-analysis-using-salib/
    print(df.ix[:,0:20].columns.tolist())
    print(len(df.ix[:,0:20].columns.tolist()), X_real.shape[1])

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]]*3
    }

    param_values = saltelli.sample(problem, 100, calc_second_order=True)
    print(param_values.shape)

    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y, calc_second_order=False)
    print(Si)

    problem = {
        'num_vars': X_real.shape[1],
        'names': df.ix[:,0:20].columns.tolist(),
        'bounds': [[-np.pi, np.pi]]*3
    }

    Si = sobol.analyze(problem, Y_real, calc_second_order=False)
    print(Si)
#sobol_analysis()


def surrogate_model(X_train, X_test, Y_train, Y_test):

    lr = LinearRegression()
    lasso = Lasso()
    rr = Ridge()
    pls = PLSRegression() # is really inaccurate, can I increase its accuracy??
    knn = KNeighborsRegressor(5, weights='uniform')
    nn = MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs')
    rf = RandomForestRegressor()

    #ransac = RANSACRegressor()
    #hr = HuberRegressor()

    # do not support multivariate regression, use MultiOutputRegressor
    bayesm = BayesianRidge()
    svrm = SVR(kernel='linear', C=1000) # takes about 100s
    gpr = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))

    #("PLS", pls), ("SVR", svrm)
    models = [("LR",lr), ("Lasso", lasso), ("rr", rr),  ("k-NN", knn), ("NN", nn), ("RF", rf), ("BR", bayesm), ("GPR", gpr)]

    x=0
    fig = plt.figure()
    ax1, ax2 = plt.subplot(211), plt.subplot(212)

    r2_list, mse_list, time_list, mae_list = [], [], [], []
    for name, model in models:
        if model in {bayesm, svrm, gpr}:
            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            #                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            # surrogate = GridSearchCV(svrm, tuned_parameters)
            model = MultiOutputRegressor(model)

        stime = time.time()
        prediction = model.fit(X_train, Y_train).predict(X_test)
        print("Time for", name, ": %.3f" %(time.time() - stime))
        time_list.append((time.time() - stime))
        r2_list.append(r2_score(Y_test, prediction, multioutput='uniform_average'))
        mse_list.append(mean_squared_error(Y_test, prediction))
        mae_list.append(mean_absolute_error(Y_test, prediction))
        print("mean-squared-error:", mean_squared_error(Y_test, prediction))

        #print "r2:", model.score(X_test, Y_test)

        if model == svrm:
            print("hi")
            print(model.score())
        prediction = prediction[prediction[:, 0].argsort()] # sort the predicted data
        ax1.plot(prediction[:,0], color=colors[x], label=name+' T1') # plot first column
        ax1.plot(prediction[:,1], color=colors[x])
        x+=1

    y_pos = np.arange(len(models))
    bar_width = .3
    print(y_pos)
    print(r2_list)
    print(mae_list)
    print(time_list)
    ax2.barh(y_pos, r2_list)
    ax2.barh(y_pos, mae_list)
    ax2.barh(y_pos, mse_list)


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*.8, box.height])
    Y_test = Y_test[Y_test[:,0].argsort()]
    ax1.plot(Y_test, 'o', color='black', label='actual')
    ax1.legend(bbox_to_anchor=(1,.5), loc='center left') #http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    #plt.tight_layout()
    plt.show()
surrogate_model(X_train, X_test, Y_train, Y_test)

# TODO sensitivity analysis / sobol indices / uncertainty decomposition
# TODO for GPR, what kernel to use?
# TODO MARS not implemented in sklearn it seems, could use pyearth https://statcompute.wordpress.com/2015/12/11/multivariate-adaptive-regression-splines-with-python/
# TODO http://scikit-learn.org/stable/modules/model_evaluation.html
# TODO http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
# TODO have to cross-validate SVR the choice of parameters to increase regression accuracy

# TODO timeseries
# http://stackoverflow.com/questions/20841167/how-to-predict-time-series-in-scikit-learn
# http://stackoverflow.com/questions/30346605/time-series-forecasting-with-scikit-learn
# http: // machinelearningmastery.com / arima - for -time - series - forecasting -with-python /
# http: // stackoverflow.com / questions / 31379845 / forecasting - with-time - series - in -python


# TODO Polynomial regression
# TODO Logistic Regression is a classifier not a regressor....
# TODO Principal component regression (not integrated in sklearn, but can be implemented using some data shuffling.)
# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11/Lab%2011%20-%20PCR%20and%20PLS%20Regression%20in%20Python.pdf
# http://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python
# http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#sphx-glr-auto-examples-plot-digits-pipe-py





# # Run this once to see which is the best fit
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# svr_bestfit = GridSearchCV(SVR(kernel='rbf', C=1, degree=3, gamma='auto', tol=0.001), tuned_parameters)
#
# svr_bestfit = svr_bestfit.fit(X_train, Y_train.ravel())
# Y_SVR_bestfit = svr_bestfit.predict(X_test)
# print svr_bestfit.best_params_, svr_bestfit.best_score_
#Results: {'kernel': 'linear', 'C': 1000} 0.873165196262
#print X_train.shape, Y_train.shape
# Support Vector Regression with linear and rbf kernel
#stime = time.time()
#Y_SVR_LR = LinearSVR().fit(X_train, Y_train).predict(X_test)
#Y_SVR_RBF = SVR(kernel='rbf', C=1000, degree=3, gamma=0.0001, tol=0.001).fit(X_train, Y_train.ravel()).predict(X_test)
# print("Time for SVR linear prediction: %.3f" % (time.time() - stime))

#Multiregressor  http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor

#model evaluation
#http://scikit-learn.org/stable/modules/model_evaluation.html


# print Y_test[:5]
# Y = np.hstack((Y_test, Y_knn)) # Y_lr, Y_ridge, Y_knn, Y_nn, Y_pls concatenate the arrays, real and predicted results
# #print 'True data\n',Y[:5]
#
# # PLOT RESULTS
# plt.figure(0)
# for i in xrange(len(Y[0])):
#     plt.plot(Y[:,i], label=i)
# plt.legend(loc='best')
#
# Y = Y[Y[:,0].argsort()] # sort the arrays based on the real results from low to high
# plt.figure(1)
# for i in xrange(len(Y[0])):
#     plt.plot(Y[:,i], label=i)
#
# #plt.title("MSE: %s\n R^2: %s" % (mean_squared_error(Y_test, Y_lr), r2_score(Y_test, Y_lr)))
# plt.legend(loc='best')
# plt.show()



#Linear Regression
# stime = time.time()
# lr = LinearRegression()
# Y_lr = lr.fit(X_train, Y_train).predict(X_test)
# print("Time for Linear Regression: %.3f" % (time.time() - stime))
# #print 'Linear Regression Coefficients\n', lr.coef_ # what is lr.coef_[0] ? the features? http://stackoverflow.com/questions/26951880/scikit-learn-linear-regression-how-to-get-coefficients-respective-features
# print 'MSE Linear Regression = ', mean_squared_error(Y_test, Y_lr)
# print 'R^2 = ', lr.score(X_test, Y_test)



# Ridge Regression
# stime = time.time()
# ridge = Ridge()
# Y_ridge = ridge.fit(X_train, Y_train).predict(X_test)
# print("Time for Ridge Regression fitting: %.3f" % (time.time() - stime))
# print 'MSE Ridge Regression = ', mean_squared_error(Y_test, Y_ridge)
# print 'R^2 = ', ridge.score(X_test, Y_test)

# Partial Least Squares Regression
# stime = time.time()
# pls = PLSRegression(X_train.shape[1]).fit(X_train, Y_train)
# Y_pls = pls.predict(X_test)
# print("Time for PLS Regression fitting: %.3f" % (time.time() - stime))
# #print 'model coefficients: \n', pls.coef_
# print 'intercepts: \n', pls.y_mean_ - np.dot(pls.x_mean_, pls.coef_)
# print 'MSE PLS =', mean_squared_error(Y_test, Y_pls)
# print 'R^2 PLS =', pls.score(X_test, Y_test)
#
# # PLSCanonical (univariate)

# ! Kernel ridge regression is really slow...
# X_plot = np.linspace(0, 20, 10000)[:, None]
# stime = time.time()
# Y_kr = kr.predict(X_test)
# print Y_kr
# print("Time for KRR prediction: %.3f" % (time.time() - stime))

#print 'MSE Support Vector Regression with RBF kernel: ', mean_squared_error(Y_test, Y_SVR_RBF)
#print 'MSE Support Vector Regression with Linear kernel: ', mean_squared_error(Y_test, Y_SVR_LR)


# k-Nearest Neighbors Regression
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py
# n_neighbors = 5
# # weights = ['uniform', 'distance']
# stime = time.time()
# knn = KNeighborsRegressor(n_neighbors, weights='uniform')
# Y_knn = knn.fit(X_train, Y_train).predict(X_test)
# print("Time for k-Nearest Neighbors: %.3f" % (time.time() - stime))
# print 'MSE KNN = ', mean_squared_error(Y_test, Y_knn)
# print 'R^2 KNN = ', knn.score(X_test, Y_test)
# print 'Y_knn', Y_knn.shape


# """ Neural Network """
# # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
# solvers = ['lbfgs', 'sgd', 'adam'] # adam is also stochastic gradient
# nn = MLPRegressor(hidden_layer_sizes=(50,), solver='lbfgs')
# print Y_train.shape
# stime = time.time()
# if Y_train.shape[1] > 1:
#     Y_nn = nn.fit(X_train, Y_train).predict(X_test) # Y needs to be in shape of y to (n_samples, ) print y.shape to see what it looks like
# else:
#     Y_nn = nn.fit(X_train, Y_train.ravel()).predict(X_test)
# print("Time for Neural Network: %.3f" % (time.time() - stime))
# print 'MSE NN = ', mean_squared_error(Y_test, Y_nn)
# print 'R^2 NN = ', nn.score(X_test, Y_test) #r2_score(Y_test, Y_nn)
# Y_nn = Y_nn[:,np.newaxis] # have to create a new axis to add dimension to the array
# print 'Y_nn',Y_nn.shape
# print Y_nn[:5]
# #scaling? http://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use



""" #It seems that the current version of sklearn does notn allow for multiple inputs to the same target output (http://stackoverflow.com/questions/34723703/gaussian-process-scikit-learn-exception)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_test, return_cov=True)
Y_gp = gp.predict(X_test)

plt.figure(0)
plt.plot(X_real, Y_real, 'k', lw=3, zorder=9)
# plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
#                  y_mean + np.sqrt(np.diag(y_cov)),
#                  alpha=0.5, color='k')
plt.plot(X_real, Y_gp, 'r', lw=3, zorder=9)
#plt.scatter(X[:, 0], y, c='r', s=50, zorder=10)
plt_show()
"""

""" # GAUSSIAN PROCESS REGRESSION
# First run
plt.figure(0)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10)
plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          % (kernel, gp.kernel_,
             gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()

# Second run
plt.figure(1)
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10)
plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          % (kernel, gp.kernel_,
             gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()
plt.show()
"""

"""
X1 = np.linspace(0,100,101)
Y1 = np.array([(100*np.random.rand(1)+num) for num in (5*X1+10)])
print X1.shape, Y1.shape
print X1

# TODO also look into principal components regression
n = 1000 # n_samples
q = 3 # n_outputs
p = 10 # n_inputs

X = np.random.normal(size=n * p).reshape((n, p)) # n_samples (n) of random n_inputs (p)
B = np.array([[1, 2] + [0] * (p - 2)] * q).T # create noise for the outputs and function to determine Y based on X values
# each Yj = 1*X1 + 2*X2 + noize
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5 # n_samples (n) of random n_outputs with noise
print X.shape, Y.shape
"""


## linear regression and cross validation with scikit learn

#Gaussian Process Modelling
# http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr

# Gaussian kernel, what is this?
# http://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# this is an NxD matrix, where N is number of items and D its dimensionalites
# sigma=5
# pairwise_dists = squareform(pdist(X, 'euclidean'))
# K = np.exp(-pairwise_dists ** 2 / sigma ** 2)
# print K

# def fn(x, a, b, c):
#     return a+ b*x[0] + c*x[1]
#
# popt, pcov = curve_fit(fn, x, y)
# print popt
# print pcov
