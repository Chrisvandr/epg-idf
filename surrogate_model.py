import numpy as np
from numpy import linspace
import getpass
import time
import pandas as pd
import os
import random
import sys
import array
import csv
import math
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D

#import scipy
from scipy.optimize import curve_fit, leastsq, least_squares, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, sem
from scipy.stats.kde import gaussian_kde
from sklearn.feature_selection import f_regression, mutual_info_regression

#import statsmodel for VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

#Sensitivity modules
from SALib.sample import saltelli
from SALib.analyze import sobol, delta, fast, morris, dgsm, ff
from SALib.test_functions import Ishigami

#Sci-kit modules
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, TheilSenRegressor, HuberRegressor, RANSACRegressor # Linear Regression / Ordinary Least Squares
from sklearn.cross_decomposition import PLSRegression # Partial Least Squares Regression
from sklearn.svm import SVR, LinearSVR #Support Vector Regression
from sklearn.kernel_ridge import KernelRidge # Kernel Ridge Regression
from sklearn.gaussian_process import GaussianProcessRegressor # Kriging/GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C # GPR kernels
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor # For Multivariate regression not implemented for some

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.externals import joblib
import inspect
import pickle

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import algorithms
from decimal import *



#own scripts
sys.path.append('C:/Users/' + getpass.getuser() + '\Dropbox/01 - EngD/07 - UCL Study/UCLDataScripts')
from PlotSubmetering import ReadRuns
from PlotSubmetering import lineno
from PlotSubmetering import ReadSubMetering

def plot_surrogate_learning_rate():
    df_r2 = pd.DataFrame()
    df_mse = pd.DataFrame()
    its = np.arange(20, len(Y_train), 20)
    for i in range(1, len(Y_train)):
        if i % 20 == 0:
            print(i)
            r2_list, mse_list = SurrogateModel(X_train[:i], X_test, Y_train[:i], Y_test, cols_outputs, cols_inputs,
                                                time_step, plot_progression=True, write_model=False, write_data=False)
            # print(df_scores)
            r2_list = pd.DataFrame(pd.Series(r2_list))
            mse_list = pd.DataFrame(pd.Series(mse_list))
            df_r2 = pd.concat([df_r2, r2_list])
            df_mse = pd.concat([df_mse, mse_list])

    if NO_ITERATIONS > 20:
        df_r2.index = its
        df_r2.columns = ['R2']
        df_r2 = df_r2.astype(float)
        df_mse.index = its
        df_mse.columns = ['MSE']
        df_mse = df_mse.astype(float)
        # df_mse.index = its
        print('r2', df_r2)
        print('mse', df_mse)

        ax = df_r2.plot(color=COLORS[0])
        ax2 = ax.twinx()
        df_mse.plot(ax=ax2, color=COLORS[1])

## PLOT Correlations ##
def plot_correlations(df_corr, cols_outputs):
    fig, ax = plt.subplots()
    width = 1/len(cols_outputs)
    df_corr = df_corr[abs(df_corr[cols_outputs[0]]) > 0.2]
    print(df_corr.index)
    print(df_corr)
    #locs, labels = xticks()
    ind = np.arange(len(df_corr))
    for i in range(len(cols_outputs)):
        ax.barh(ind+i*width, df_corr[cols_outputs[i]], width, label=cols_outputs[i])
    ax.set_yticks(ind+.5) # set positions for y-labels, .5 to put the labels in the middle
    ax.set_yticklabels(df_corr.index) # set y labels ('variables') from index
    ax.set_xlabel('Correlation coefficient')
    box = ax.get_position()
    ax.legend(bbox_to_anchor=(1, .5), loc='center left') # bbox_to_anchor=(all the way next to the plot =1 if center left = loc, height 0.5 is in the middle)
    plt.tight_layout(rect=[0, 0, 0.8, 1], pad=.75, w_pad=0.1, h_pad=0.1) # use rect to adjust the plot sides rects=[left, bottom, right, top]
    # plt.xlabel('Correlation')
    # plt.yticks(y_pos + bar_width / 2, spearman_results.index)
    # plt.legend()

## PLOT HEATMAP ##
def HeatmapCorrelations(df, cols_outputs):
    #df_corr = df_corr[abs(df_corr[cols_outputs[0]]) > 0.3] # based on first columns
    #df_corr = df_corr[df_corr > .5]
    #df = df[df[df > 10].any(1)] # select rows where any column is greater than 10
    #df = df[abs(df[abs(df) > .25].count(axis=1) > 0.35)] # where any value is greater than .5 and any column is greater than 1
    #todo perhaps exclude the server_exclusive input as it is the strongest correlator...)
    #df = df[abs(df['Cooling'] > .25)]
    #df = df[abs(df.iloc[:,1] > .25)]
    #vif_indices = [i for i, v in enumerate(vif) if v == np.inf or v < 6]  # when to exclude features

    heatmap = df.as_matrix(columns=cols_outputs)
    #print(heatmap)
    #cm = plt.get_cmap('spectral')
    fig, ax = plt.subplots()

    #ax = sns.heatmap(heatmap)
    #ax.set_aspect("equal")

    #plot
    im = ax.matshow(heatmap, cmap=cm.Spectral_r, interpolation='none')
    ind_x = np.arange(df.shape[1])
    ind_y = np.arange(df.shape[0])
    print(ind_x, ind_y)

    ax.set_aspect('equal')
    ax.set_yticks(ind_y)  # set positions for y-labels, .5 to put the labels in the middle
    ax.set_yticklabels(df.index)  # set y labels ('variables') from index
    ax.set_yticks(ind_y + .5, minor=True)
    ax.set_xticklabels('')
    ax.set_xticks(ind_x)  # set positions for y-labels, .5 to put the labels in the middle
    ax.set_xticklabels(cols_outputs, rotation=90)  # set y labels ('variables') from index
    ax.set_xticks(ind_x + .5, minor=True)
    ax.set_aspect('equal')

    ax.grid(which='minor', linewidth=1, color='white')
    ax.grid(False, which='major')
    #ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # annotating the data inside the heatmap
    for y in range(df.shape[0]):
        for x in range(df.shape[1]):
            plt.text(x, y, '%.2f' % heatmap[y][x],horizontalalignment='center',verticalalignment='center',fontsize=6)

    cbar = plt.colorbar(im, fraction=0.04555, pad=0.04)
    cbar.ax.tick_params()
    #ax.legend(bbox_to_anchor=(1, .5), loc='center left')  # bbox_to_anchor=(all the way next to the plot =1 if center left = loc, height 0.5 is in the middle)
    plt.tight_layout(rect=[0, 0, 0.8, 1], pad=.75, w_pad=0.1, h_pad=0.1)  # use rect to adjust the plot sides rects=[left, bottom, right, top]
# PLOT: Plot predictions based on best individuals in each generation
def plot_prediction_gen():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    gen = logbook.select("gen")
    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i])
        result.append(prediction)
        #print('best indices used array:', best_inds[i])
        #print('best indices used for prediction', prediction)

    for i,v in enumerate(result):
        alp = 1-(i/50)
        ax.plot(v, label=gen[i], alpha=alp)
    ax.plot(targets, 'o', color='black', label='Target')
    ax.legend(loc='best')
    ax.set_title('ax2, best individual per generation')

# PLOT: Plot energy use for the objectives with increasing generation
def plot_energy_objective(best_inds, cols_outputs, targets):
    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i])
        result.append(prediction)

    df_result = pd.DataFrame(result, columns=cols_outputs)
    ax = df_result.plot(title='Prediction of the best individual per generation', color=COLORS)
    targets_dup = ([targets, ]) * len(result)  # duplicate targets list
    df_target = pd.DataFrame(targets_dup, columns=cols_outputs)
    df_target.plot(ax=ax, style='--', color=COLORS)

# PLOT: Plot objective differences using RMSE
def plot_best_fit(cols_outputs, best_inds_fitness):
    #df_avg = pd.DataFrame(fit_avgs, columns=cols_objectives)
    #df_mins = pd.DataFrame(fit_mins, columns=cols_objectives)
    df_inds_fit = pd.DataFrame(best_inds_fitness, columns=cols_outputs)

    ax = df_inds_fit.plot(color=COLORS)
    ax.set_title("Best Individual fitnesses per gen")

# PLOT: Plot objective differences using RMSE
def plot_fit():
    #df_avg = pd.DataFrame(fit_avgs, columns=cols_objectives)
    df_mins = pd.DataFrame(fit_mins, columns=cols_outputs)
    #df_inds_fit = pd.DataFrame(best_inds_fitness, columns=cols_objectives)

    ax = df_mins.plot(color=COLORS)
    ax.set_title("Minimal fitnesses over population per gen")
def test_data(DataPath):
    df = pd.read_csv(DataPath+ 'test_data.csv', header=0)

    X_real = df.ix[:,0:20]
    cols_inputs = X_real.columns.tolist()
    X_real = X_real.as_matrix()
    Y_real = df.ix[:,21:23]
    cols_outputs = Y_real.columns.tolist()
    Y_real = Y_real.as_matrix()

    # X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real) #randomly split your data
    # print(X_train.shape, Y_train.shape)
    # print(Y_train.ravel().shape)
    # print(X_real)

    return Y_real, X_real, cols_outputs, cols_inputs

def Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step):
    #fig = plt.figure(0)
    #ax = plt.subplot(111)

    print(lineno(), 'no. variables', len(cols_inputs))
    print(spearmanr(X_real[:,0], Y_real[:,0])[0])
    print(pearsonr(X_real[:,0], Y_real[:,0])[0])

    df_corr = pd.DataFrame(cols_inputs)
    df_corr.columns = ['variables']
    for j in range(Y_real.shape[1]):
        spearman_list, pearsonr_list = [], []
        for i in range(X_real.shape[1]):
            spearman_list.append(spearmanr(X_real[:, i], Y_real[:, j])[0])
            pearsonr_list.append(pearsonr(X_real[:, i], Y_real[:, j])[0])
        #append list to df
        spear = pd.Series(spearman_list)
        df_corr[cols_outputs[j]] = spear

    df_corr.set_index('variables', inplace=True)
    df_corr.to_csv(DataPath_model_real + 'Correlations' + time_step + '.csv')
    print(df_corr.shape)

    ## PLOT Check convergence of Correlations coefficients over model iterations ##
    plot_std = 0
    if plot_std is 1:
        df_stdcorr = pd.DataFrame()
        output_variable = 4 # which output variable (energy use) to look at
        for i in range(101, 110, 1): # looking at different input variables
            df_std  = pd.DataFrame()
            spearman_list = []
            iterations = []
            for j in range(9, 300, 10):
                iterations.append(j)
                spearman_list.append(spearmanr(X_real[:j, i], Y_real[:j,output_variable])[0]) #last 0 signifies the rho value (1 = p-value)
            s1 = pd.Series(spearman_list) # put Correlations in series
            df_std = pd.concat([df_std, s1], axis=0) #combine empty dataframe with Correlations from series
            df_std.columns = [cols_outputs[output_variable]+'_'+cols_inputs[i]] #name column
            df_stdcorr = pd.concat([df_stdcorr, df_std], axis=1) #combine multiple Correlations

        s_it = pd.Series(iterations, name='iterations')
        df_stdcorr = pd.concat([df_stdcorr, s_it], axis=1)
        df_stdcorr.set_index('iterations', inplace=True)

        ax1 = df_stdcorr.plot(title='Spearman Correlation')
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * .6, box.height])

        ax1.legend(bbox_to_anchor=(1, .5), loc='center left')
        ax1.set_ylabel('Correlation')
        ax1.set_xlabel('Iterations')

    #spearman_results.columns = ['correlation']  # rename columns
    #spearman_results = spearman_results.sort_values(['correlation'], ascending='True')
    #print(spearman_results.head())

    # PLOT with Matplotlib
    # y_pos = np.arange(X_real.shape[1])
    # bar_width = .8
    # ax.barh(y_pos, spearman_results['correlation'], bar_width,color='#1f77b4',label='Spearman'+cols_outputs[0])
    # ax.barh(y_pos+bar_width, pearsonr_list, bar_width,
    #         color='#ff7f0e',
    #         label='Pearson')

    #spearman_results = pd.DataFrame(pd.Series(df_corr, cols_inputs))
    #df_corr.iloc[:,0].plot.barh(stacked=True)
    return df_corr

def ImportOutputs(DataPath_model_real, inputs, run_numbers, feature_selection):
    df = pd.read_csv(DataPath_model_real + inputs, header=0) #TODO using backup file atm
    cols_inputs = df.columns.tolist()

    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)
    X_real = df.ix[run_numbers]

    X_real = X_real.drop(df.columns[[0]], axis=1) #drop run_no column
    #print(lineno(), X_real.head(2))
    X_real = X_real.as_matrix()

    # FEATURE SELECTION

    #  univariate selection
    if feature_selection is True:
        univariate_selection = True
        if univariate_selection is True:
            from sklearn.feature_selection import VarianceThreshold
            sel = VarianceThreshold(threshold=(.8 * (1 - 0.8)))
            sel.fit(X_real) # fit the inputs
            selected_feature_indices = sel.get_support(True) #get indices of those selected
            print(lineno(), 'selected features', len(selected_feature_indices), selected_feature_indices)
            X_real = sel.transform(X_real) #transform the matrix
            cols_inputs = [cols_inputs[i] for i in selected_feature_indices.tolist()] # select the names based on selected features
            print(lineno(), cols_inputs)

        # VIF selection
        VIF_selection = True
        if VIF_selection is True:
            X_vars = np.append(X_real, np.ones([len(X_real), 1]), 1) # need to add a column of ones at the end...
            vif = [variance_inflation_factor(X_vars, i) for i in range(X_vars.shape[1])] # Catch infs if not enough runs
            print(lineno(), len(vif), vif)
            vif_indices = [i for i, v in enumerate(vif) if v == np.inf or v < 6] # when to exclude features
            print(lineno(), 'vif indices', vif_indices)
            vif_indices = vif_indices[:-1] ## need to remove the last item as it is an added zero value.
            if len(vif_indices) > 0:
                print(lineno(), X_real.shape)
                X_real = X_real[:, vif_indices]
                cols_inputs = [cols_inputs[i] for i in vif_indices]

    #print(X_real.shape)
    #print(lineno(), 'Variance Influence Factors', vif)

    # scale the data?
    #todo http://scikit-learn.org/stable/modules/preprocessing.html

    return X_real, cols_inputs

# Run the surrogate learning process with different sample sizes to see the learning rate.
def SurrogateModel(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, plot_progression, write_model, write_data):
    #todo check multicolinearity of variables https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels
    # todo http://scikit-learn.org/stable/modules/feature_selection.html
    #todo https://etav.github.io/python/vif_factor_python.html
    #when some variables are highly correlated it makes the model unstable.
    print(lineno(), 'Xtrain, Ytrain', X_train.shape, Y_train.shape)

    lr = LinearRegression()
    lasso = Lasso()
    rr = Ridge()
    pls = PLSRegression() # is really inaccurate, can I increase its accuracy?? # is supposedly very good for dealing with multicollinearity among X values and suited when the matrix of predictors has more variables than observations
    knn = KNeighborsRegressor(5, weights='uniform')
    nn = MLPRegressor(hidden_layer_sizes=(5, 5), solver='lbfgs', activation='relu', random_state=1, max_iter=500, power_t=.5, tol=.0001, learning_rate='constant') #TODO scale data for SVR and also NN #http://scikit-learn.org/stable/modules/svm.html#regression
    rf = RandomForestRegressor()
    ts = TheilSenRegressor()
    ransac = RANSACRegressor()
    hr = HuberRegressor()

    # do not support multivariate regression, use MultiOutputRegressor
    bayesm = BayesianRidge()
    svrm = SVR(kernel='rbf', C=1) # takes about 100s #TODO scale data for SVR and also NN #http://scikit-learn.org/stable/modules/svm.html#regression
    gpr = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))

    #("PLS", pls), ("SVR", svrm), # take too long
    #("rr", rr),  ("k-NN", knn), ("NN", nn), ("RF", rf), ("BR", bayesm), ("Lasso", lasso), ("GPR", gpr), ("LR", lr)
    #("ts", ts), ("ransac", ransac), ("hr", hr) perform as well as Linear Regression / OLS
    models = [("rr", rr)] # is of type list #print(type(models))
    #todo even though LR is quite accurate, it will give very high coefficient as it is overfitting the model, specifically for the overtime_multiplier inputs, RR seems to give more stable values.
    x=0

    y_pos = np.arange(len(models)) # no. of models/methods i am using.
    r2_list, mse_list, time_list, mae_list, evar_list = [], [], [], [], [] # fill lists with evaluation scores

    predictions = pd.DataFrame()
    for i in range(len(cols_outputs)):
        true_series = pd.Series(Y_test[:,i], name=cols_outputs[i]+'_true')
        predictions = pd.concat([predictions, true_series], axis=1)
    model_names = []

    print(lineno(), 'columns', cols_outputs)
    for name, model in models:
        model_names.append(name)
        stime = time.time()
        if model in {bayesm, svrm, gpr}: # use multioutputregressor for these methods
            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            #                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            # surrogate = GridSearchCV(svrm, tuned_parameters)
            model = MultiOutputRegressor(model)

            ## Fitting
            print(name)
            model.fit(X_train, Y_train)
            prediction = model.predict(X_test)

            if write_model == True:
                joblib.dump(model, DataPath_model_real+name + '_' + time_step + '_model.pkl')
            else:
                print(lineno(), 'write_model set to:', write_model)

        elif model in {lr, rr, nn, knn, rf, lasso, nn, pls}:
            if model == nn:
                scaler = StandardScaler()
                scaler.fit(X_train)
                StandardScaler(copy=True, with_mean=True, with_std=True)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                model.fit(X_train, Y_train)
                prediction = model.predict(X_test)
            else:
                model.fit(X_train, Y_train)
                prediction = model.predict(X_test)

            if write_model == True:
                joblib.dump(model, DataPath_model_real+name + '_' + time_step + '_model.pkl')
            else:
                print(lineno(), 'write_model set to:', write_model)

        # timing / scoring
        mse_day, mae_day = [], []

        raw_mse = mean_squared_error(Y_test, prediction, multioutput='raw_values')
        raw_mae = mean_absolute_error(Y_test, prediction, multioutput='raw_values')  # multioutput='raw_values' will give the error for all 31days*8enduses (248, )

        print(lineno(), 'raw_mse', raw_mse.shape)
        for i in range(8):
            mse_day.append(np.sum(raw_mse[i::8])) # sum mse of all 31 days for every end-use.
            mae_day.append(np.sum(raw_mae[i::8]))
        total_mae = np.sum(mae_day)

        print(lineno(), 'mse_day', mse_day)
        print(lineno(), 'mae_day', mae_day)
        print(lineno(), 'total_mae', total_mae)
        print(lineno(), len(mse_day))

        time_list.append("{0:.4f}".format(((time.time() - stime))))
        r2_list.append("{0:.4f}".format((r2_score(Y_test, prediction, multioutput='uniform_average'))))
        mse_list.append("{0:.4f}".format((mean_squared_error(Y_test, prediction))))
        mae_list.append("{0:.4f}".format(total_mae))
        evar_list.append("{0:.4f}".format((explained_variance_score(Y_test, prediction))))

        print(lineno(),'no. of predictions', len(prediction), prediction)
        print(lineno(), 'mae', mean_absolute_error(prediction, Y_test))

        #raw_mse = mean_squared_error(Y_test, model.predict(X_test), multioutput='raw_values')
        #print("{0:.4f}".format((mean_squared_error(Y_test, prediction))))

        abs_diff = [(j - i)**2 for i, j in zip(Y_test, prediction)]
        mse_abs_diff = [np.mean(i)/len(Y_test) for i in abs_diff] # the mean squared error for each iteration separately

        if end_uses is True:
            abs_perc_error = [abs((i - j)/i)*100 if i > 0 else 0 for i, j in zip(Y_test[1,:], prediction[1,:])]
            abs_perc_error_enduse = []
            for i in range(8):
                abs_perc_error_enduse.append(np.mean(abs_perc_error[i::8])) # sum mse of all 31 days for every end-use.

            print(lineno(), 'single prediction', prediction[1,:].tolist())
            print(lineno(), 'single test data', Y_test[1,:].tolist())
            print(lineno(), 'single abs perc error per day per end use', abs_perc_error)
            print(lineno(), 'mean abs perc error per end use', abs_perc_error_enduse)
            #print(len(abs_perc_error))
            #print(mse_abs_diff)
            #print(len(abs_diff[0]))

        raw_rmse = []
        for mse in raw_mse:
            raw_rmse.append(np.sqrt(mse)*1000)

        # Get coefficients and intercepts, this actually works different for each algorithm
        if model in {bayesm, svrm, gpr}:
            print(lineno(), 'model score', models[x][1].score)
            if name == 'SVR':
                df_coef = pd.DataFrame(models[x][1].coef_, cols_outputs)
                df_intercept = pd.DataFrame(models[x][1].intercept_, cols_outputs)
                #print('params', models[x][1].get_params)
        elif model in {lr, rr, nn, knn, rf, lasso, nn, pls}:
            df_coef = pd.DataFrame(models[x][1].coef_, cols_outputs)
            df_intercept = pd.DataFrame(models[x][1].intercept_, cols_outputs)


        ## Send function coefficients and intercept for each model to csv
        df_coef.to_csv(DataPath_model_real + name +'_coef_' + time_step + '.csv')
        df_intercept.to_csv(DataPath_model_real + name + '_intercept_' + time_step + '.csv')

        ## Scatterplot of the test data and prediction data for one single iteration
        #TODO combine the diff prediciont from multiple algorithms
        # pred_series = pd.DataFrame(prediction[1, :])
        # test_series = pd.DataFrame(Y_test[1, :])
        # df_pred_test = pd.concat([pred_series, test_series], axis=1)
        # df_pred_test.columns = ['prediction', 'test']
        # print(df_pred_test)
        # df_pred_test.plot(kind='scatter', x='prediction', y='test')

        if plot_progression == False:
            fig = plt.figure()
            ax1 = plt.subplot(111)
            for i in range(len(cols_outputs[:9])):
                prediction_series = pd.Series(prediction[:,i], name=cols_outputs[i]+'_'+name)
                predictions = pd.concat([predictions, prediction_series], axis=1)

                prediction_std = np.std(predictions)
                #print(cols_outputs[i], name, prediction[:5], Y_test[:5])
                ax1.plot(prediction[:,i], color=COLORS[x], label=name if i == 0 else "") # plot first column, and label once

        if write_data == True:
            predictions.to_csv(DataPath_model_real + name +'_predictions_' + time_step + '.csv')
        x+=1

    #print(Y_test.shape)
    if plot_progression == False:
        for i in range(len(cols_outputs[:9])):
            ax1.plot(Y_test[:, i], 'o', color=COLORS[i], label=cols_outputs[i])

    #Y_test = Y_test[Y_test[:,0].argsort()] # sort test data based on frist column

    #print('time [s]:', time_list)
    df_scores = pd.DataFrame(np.column_stack([[x[0] for x in models], mae_list]),
                             columns=['Algorithms','Mean absolute error']) #, 'Mean squared error $\mathregular{[kWh/m^{2}a]}$'
    df_r2 = pd.DataFrame(np.column_stack([[x[0] for x in models], r2_list]), columns=['Algorithms', '$\mathregular{r^{2}}$'])
    df_scores.set_index('Algorithms', inplace=True)
    df_r2.set_index('Algorithms', inplace=True)
    df_scores=df_scores.astype(float) #change data back to floats..
    df_r2 = df_r2.astype(float)

    if plot_progression == False:
        ax = df_scores.plot(kind='bar', stacked=False, width=.15, position=0)
        ax2 = ax.twinx()
        df_r2.plot(ax=ax2, kind='bar', width=.15, position=1, color=COLORS[5])

        print(lineno(), 'learning times [seconds]', time_list)

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width*.8, box.height])

        ax1.legend(bbox_to_anchor=(1,.5), loc='center left')    #http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        ax1.set_ylabel('[kWh/m2a]')
        #plt.tight_layout()

    return r2_list, mse_list

def CreateElements(mus, sigmas):  # create input variables
    # elements = [random.uniform(a, b) for a, b in zip(lower_bound, upper_bound)]
    elements = []
    for x in range(len(mus)):
        elements.append(random.gauss(mus[x], sigmas[x]))  # set to 10% variation currently... #TODO change
    # print([float(Decimal("%.2f" % e)) for e in elements])
    # data = [float(Decimal("%.2f" % e)) for e in mylist]
    # myRoundedList = [round(elem, 2) for elem in myList]
    # print([float(Decimal("%.2f" % e)) for e in df_inputs.iloc[0, :df_coefs.shape[1]].tolist()])
    return elements

def calculate(individual):  # predict using input variables and pkl model
    model = joblib.load(DataPath_model_real + 'rr_' + time_step + '_model.pkl')
    individual = np.array(individual).reshape((1, -1))

    prediction = model.predict(individual)[0]
    # print(prediction)
    return prediction

def EvaluateObjective(individual):
    diff = []
    prediction = calculate(individual)
    prediction = [round(i,0) for i in prediction]

    for y in range(len(targets)):
        output = math.sqrt((targets[y] - prediction[y]) ** 2)
        #output = abs(targets[y] - prediction[y])
        diff.append(output)

    #todo does this actually work correctly?

    #normalize output
    #s = sum(diff)
    #dif f = [float(i)/s for i in diff]
    #print(['targets', 'prediction', 'difference'])
    #print(targets, prediction, diff)
    #diff = diff[:NO_OF_OBJECTIVES]
    #diff = [float(i)/sum(diff) for i in diff]
    #print(tuple(diff))
    return tuple(diff)

# Load the pickle surrogate model made by 'PKL_SurrogateModel.py' and calculate with new inputs
def PKL_SurrogateModel(samples, mus, sigmas, cols_outputs, cols_inputs):
    individuals = []
    predictions = []
    elements = CreateElements(mus, sigmas)

    for x in range(samples):
        individual = CreateElements(mus, sigmas)
        individuals.append(individual)
        prediction = calculate(individual)
        predictions.append(prediction)
        if x % 10 == 0:
            print(x)

    df_predicted = pd.DataFrame(predictions)
    df_population = pd.DataFrame(individuals, columns=cols_inputs)

    Y_real = df_predicted.as_matrix()
    X_real = df_population.as_matrix()

    print('X_real and Y_real shapes', X_real.shape, Y_real.shape)

    return (X_real, Y_real, df_predicted, df_population)

# Sobol does not work for the test data as it is sampled by LHS and it needs to be sampled by saltelli sampling (which creates many more variables), however it can be done with the surrogate model instead.
def MetaModel_SensitivityAnalysis(time_step, cols_inputs, cols_outputs, inputs_basecase, mus, sigmas):
    # https://media.readthedocs.org/pdf/salib/latest/salib.pdf | http://salib.github.io/SALib/ # http://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    # http://keyboardscientist.weebly.com/blog/sensitivity-analysis-with-salib

    norm_bounds = pd.concat([pd.DataFrame(mus), pd.DataFrame(sigmas)], axis=1)
    norm_bounds = norm_bounds.values.tolist()  # turn into list [[mu, sigma], [mu, sigma] ...]

    bound = [float(i) * 3 for i in sigmas]
    lower_bound = [i - j for i, j in zip(mus, bound)]
    upper_bound = [i + j for i, j in zip(mus, bound)]

    # TODO determine which parameters have a small influence on outputs and use meta-model order reduction by disregarding them. Less samples will be needed for Sobol.
    D = len(cols_inputs)  # no. of parameters
    N = 1  #
    second_order = True
    if second_order == True:
        samples = N * (2 * D + 2)
    else:
        samples = N * (D + 2)
    print('no of samples', samples)
    print(D * N, 'for Fast, Delta and DGSM')  # (N*D) = [Fast, Delta, DGSM]
    print((D + 1) * N, 'for Morris')  # (D + 1)*N = [Morris, ]
    print((D + 2) * N, 'for Sobol first order')
    print(N * (2 * D + 2), 'for Sobol second order')  # , (D+2)*N for first order = [Sobol, ]
    print(2 ** (round(np.sqrt(150), 0)), 'for fractional fast')  # (2**D) < N = [FF, ] (so if 300paramaters N = 2^9 > 300)

    ## DUMMY SOBOL ##
    # def test_sobol():
    #     no_vars = 4
    #     col_names = ['x1', 'x2', 'x3', 'x4']
    #     problem = {'num_vars': no_vars,'names': col_names,'bounds': [[-np.pi, np.pi]] * no_vars}
    #     param_values = saltelli.sample(problem, 400, calc_second_order=True)
    #     print('param_values:', param_values.shape, param_values[:5])
    #
    #     Y = Ishigami.evaluate(param_values)
    #     print('Y shape:', Y.shape, Y[:5])
    #
    #     print(problem)
    #     Si = sobol.analyze(problem, Y, calc_second_order=True)
    #     print(Si['ST'])
    #     print("x1-x2:", Si['S2'][0, 1])  # interactive effects between x1 and x2.
    # #test_sobol()

    real_cols = cols_inputs  # columns names of the parameters
    problem = {'num_vars': D,
        'names': real_cols,
        'bounds': norm_bounds * D  # bound set to the input parameters
    }


    ## TODO have to use Saltelli sampling because otherwise it uses just randomised gauss, and the likely error rates would be higher..
    ## http://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis

    X_real, Y_real, df_predicted, df_population = PKL_SurrogateModel(samples, mus, sigmas, cols_outputs, cols_inputs)
    print('Sobol problem', problem)

    ## Analyze Sobol
    # If calc_second_order is False, the resulting matrix has N * (D + 2) rows, where D is the number of parameters. If calc_second_order is True, the resulting matrix has N * (2D + 2) rows.
    Si_sobol = sobol.analyze(problem, Y_real[:, 0], calc_second_order=second_order)
    # TODO sobol is analysed for a single output here. Each output has then x no. of variable correlations.

    # Si_delta = delta.analyze(real_problem, X_real, Y_real[:,0])
    # print(Si_delta['S1'])
    # print(Si_delta['delta'])

    # Si_fast = fast.analyze(real_problem, Y_real[:,0])
    #  Output sample must be multiple of D (parameters)
    # print(Si_fast['S1'])

    # Si_morris = morris.analyze(real_problem, X_real, Y_real[:,0])
    #  Output samplmes must be multiple of D+1 (parameters)
    # print(Si_morris['mu'])

    # Si = dgsm.analyze(real_problem, X_real, Y_real[:,0])
    # print(Si['delta']) # Returns a dictionary with keys ‘delta’, ‘delta_conf’, ‘S1’, and ‘S1_conf’

    # Si_ff = ff.analyze(real_problem, X_real, Y_real[:,0])
    # The resulting matrix has D columns, where D is smallest power of 2 that is greater than the number of parameters.
    # print(Si_ff['ME'])

    fig = plt.figure(0)
    ax = plt.subplot(111)
    bar_width = .45
    y_pos = np.arange(D)
    # print(y_pos)

    ax.barh(y_pos, Si_sobol['S1'], bar_width, label='First-order')
    ax.barh(y_pos + bar_width, Si_sobol['ST'], bar_width, label='Total order')

    print(cols_inputs[0], '-', cols_inputs[1], Si_sobol['S2'][0, 1])

    plt.xticks(y_pos + bar_width / 2, real_cols, rotation='vertical')
    plt.tight_layout()
    plt.legend()

def OptimisationAlgorithm(toolbox):
    random.seed(20)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)
    stats.register('std', np.std, axis=0)
    stats.register('avg', np.average, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "inputs"  # , "max", "avg" #'inputs', 'std', 'avg', 'evals'

    # Create an initial population of size n.
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.ParetoFront()

    # Test fitness values of an individual
    # ind_test = toolbox.individual()
    # ind_test.fitness.values = evaluate(ind_test)
    # print('ind_test', ind_test.fitness.values)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    pop = toolbox.select(pop, POPULATION_SIZE)  # no actual selection is done
    best_inds, best_inds_fitness = [], []
    record = stats.compile(pop)
    logbook.record(**record)  # , inputs=best_inds_fitness
    hof.update(pop)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        # print(pop), print(len(pop[0]))
        # print([ind.fitness.valid for ind in pop if ind.fitness.valid])

        offspring = tools.selTournamentDCD(pop, len(pop))  # only works with "select" to NSGA-II
        # offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]  # Clone the selected individuals

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):  # Crossover and mutate offspring
            # print('inds', ind1, ind2)
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)  # crossover randomly chosen individual within the population

                # toolbox.mutate(ind1)
                # toolbox.mutate(ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for mutant in offspring:
            if random.random() <= MUTPB:  # which offspring are mutated
                # mutate a subset of the attributes (variables) in the individuals, based on a percentage of the pop
                # for i in range(round(len(pop[0])*PERC_ATTR_TO_MUTATE)):
                toolbox.mutate(mutant)  # mutates several attributes in the individual
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # print(len(invalid_ind))
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, len(offspring))  # todo, length of offspring or pop?

        # fits = [ind.fitness.values[0] for ind in pop]
        best_ind = tools.selBest(pop, 1)[0]
        best_inds.append(best_ind)  # add the best individual for each generation
        best_inds_fitness.append(best_ind.fitness.values)

        record = stats.compile(pop)
        logbook.record(gen=gen, inputs=[int(e) for e in best_ind.fitness.values], **record)
        hof.update(pop)
        if gen % 20 == 0:
            print(gen, int(sum(targets) - sum(calculate(best_inds[-1]))), 'best_inds', [int(e) for e in calculate(best_inds[-1]).tolist()], 'targets', targets, 'fitness', [int(e) for e in best_ind.fitness.values])
            # with open("logbook.pkl", "wb") as lb_file: pickle.dump(logbook, lb_file)

    # best_ind = tools.selBest(pop, 1)[0] # best_ind = max(pop, key=lambda ind: ind.fitness)
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # print(logbook.stream)

    return pop, hof, logbook, best_inds, best_inds_fitness


def OptimiseModel(time_step, targets, mus, sigmas, cols_outputs):

    print(lineno(), 'no. of objectives', len(targets))

    bound = [float(i) * 3 for i in sigmas]
    lower_bound = [i - j for i, j in zip(mus, bound)]
    upper_bound = [i + j for i, j in zip(mus, bound)]
    elements = CreateElements(mus, sigmas)

    print('weights normalized to target', tuple([-(i) / sum(targets) for i in targets]))
    print('equal weights', tuple([-1.0 for i in targets]))
    creator.create('Fitness', base.Fitness, weights=tuple([-(i) / sum(targets) for i in targets]))
    creator.create('Individual', array.array, typecode='d', fitness=creator.Fitness)  # set or list??

    toolbox = base.Toolbox()
    ## using custom function for input generation
    toolbox.register('expr', CreateElements, mus, sigmas)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # TODO there needs to be constraint handling on the input parameters, because it's more realistic if some inputs don't change at all. Or for example the office heating setpoint seems to change to 31, which makes no sense in reality...
    # Constraint handling _> http://deap.gel.ulaval.ca/doc/dev/tutorials/advanced/constraints.html

    # TODO try different selection criteria/mutate/mate
    toolbox.register('mate', tools.cxTwoPoint)
    # toolbox.register("mate", tools.cxUniform, indpb=INDPB)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_bound, up=upper_bound, eta=20.0)

    # toolbox.register('mutate', tools.mutFlipBit, indpb=INDPB)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bound, up=upper_bound, eta=20.0, indpb=INDPB) #TODO can't divide by zero, need to remove variables that are 0
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigmas, indpb=INDPB)  # sigmas are set to a sequence, #TODO retrieve the right values, they are now based on 1/10 of some initial sample

    toolbox.register('select', tools.selNSGA2)
    # toolbox.register('select', tools.selSPEA2)

    toolbox.register('evaluate', EvaluateObjective)  # add the evaluation function
    # toolbox.register("evaluate", benchmarks.zdt1)

    pop, hof, logbook, best_inds, best_inds_fitness = OptimisationAlgorithm(toolbox)

    logbook.chapters["fitness"].header = "min", "max", "avg"

    # TODO when tolerance is achieved (say 5% within result), i can calculate for example the CV(RMSE) by combining all results from a generation with the target objectives.
    # TODO can I figure out how far the optimised result input values differ from the base input values?
    # TODO Because the predicted data for CH for example in its solution space does not include the measured data point for systems, but is still able to optimise.
    # TODO http://deap.readthedocs.io/en/master/api/tools.html#deap.tools.ParetoFront
    # TODO plot best fitness in each generation. (which may sometimes be worse than the next gen)

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    # print('best inds', best_inds)
    # sorted(individuals, key=attrgetter("fitness"), reverse=True)

    # plot:
    # http://deap.readthedocs.io/en/master/tutorials/basic/part3.html
    print('HOF', len(hof))
    front = np.array([ind.fitness.values for ind in pop])

    # print(front)

    print(best_inds[-1])
    print('cols_targets', cols_outputs)
    print('best individual prediction vs. targets')
    print('best individual', [int(e) for e in calculate(best_inds[-1]).tolist()])
    print('targets', targets)
    print('absolute diff', [i - j for i, j in zip([int(e) for e in calculate(best_inds[-1]).tolist()], targets)])

    # print(logbook)
    # print(pop)

    return best_inds, best_inds_fitness

def main():

    if build_surrogate_model is False: NO_ITERATIONS = 1 # need col names...

    X_real, cols_inputs = ImportOutputs(DataPath_model_real, inputs, run_numbers, feature_selection)  # real input data
    df_pred, runs, df_runs, df_weather = ReadRuns(building, base_case, floor_area, building_harddisk, building_abr, time_step, write_data, NO_ITERATIONS)
    cols_outputs = runs.columns.tolist()

    df_inputs = pd.read_csv(DataPath_model_basecase + inputs_basecase, header=0)
    df_inputs = df_inputs[cols_inputs] # select only those after feature selection filter!
    mus = df_inputs.iloc[0, :].values.tolist()
    sigmas = [i*(1/20) for i in mus]


    if build_surrogate_model is True:
        # Import data from parallel simulations and real data.

        Y_real = runs.as_matrix()
        print(lineno(), Y_real[:2])

        # Test data for testing algorithms.
        X_real_data = np.vstack((cols_inputs, X_real))
        Y_real_data = np.vstack((cols_outputs, Y_real))
        X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real)  # randomly split data to create test set

        ### BUILDING THE SURROGATE MODEL USING TRAINING DATA
        SurrogateModel(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, plot_progression, write_model, write_data)

        ### SENSITIVITY ANALYSIS OF REAL DATA
        # df_corr = Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step)
        # print(lineno(), df_corr.head())
        # plot_correlations(df_corr, cols_outputs)
        # HeatmapCorrelations(df_corr, cols_outputs)
        # plot_surrogate_learning_rate()

        feature_scores = []
        for i in range(Y_real.shape[1]):
            f_test, _ = f_regression(X_real, Y_real[:, i])
            feature_scores.append(f_test)
            # print('f test', f_test)
        df_feature_scores = pd.DataFrame(list(map(list, zip(*feature_scores))), columns=cols_outputs, index=cols_inputs)  # transpose list of lists and make a dataframe of it
        # df_feature_scores.plot(kind='bar', stacked=True)

        input_outputs = np.hstack((X_real_data, Y_real_data))
        if write_data is True:
            with open(DataPath_model_real + 'input_outputs_' + time_step + '.csv', "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(input_outputs)

    if metamodel_sensitivity is True:
        #### META-MODEL SENSITIVITY ANALYSIS
        MetaModel_SensitivityAnalysis(time_step, cols_inputs, cols_outputs, inputs_basecase)

        # elif time_step == 'month':
        # # todo WARNING, targets are starting from Sept-Dec then Jan-Apr (sept-dec are from 16, not 17, whereas sim files are all 16, so turn around manually for now)
        # if end_uses == True:
        #     targets = [40758, 35495, 7531, 36286, 33060, 5212, 42278, 36573, 5103, 42671, 32195, 3512, 25513, 27644, 3992, 26905, 31679, 4119, 30610, 36493, 5532, 39521, 32003, 9911]
        # elif end_uses == False:
        #     # [jan, feb, mar, apr, sep, oct, nov, dec]
        #     targets = [83785, 74560, 83954, 78379, 57150, 62704, 72635, 81437]
        #
        # #todo why are the prediction when running the surrogate model so different from the targets??? something going wrong here i think or is this just the range it can predict in, it is from a single run.
        # singleprediction[92240.45006702337, 78637.6605726759, 81411.5389778027, 72367.14122711583, 40051.313988098904, 56626.15939919354, 73364.38556002386, 73735.20007448267]
        # singletestdata[90511.33516526977, 77031.2684885664, 79573.42786600048, 70828.36384547835, 39466.62852737905, 55644.990233075674, 72172.93111136527, 73036.75992940296]
    if function_optimisation is True:
        print('optimising')
        OptimiseModel(time_step, targets, mus, sigmas, cols_outputs)

        # plot_prediction_gen()
        plot_energy_objective(best_inds, cols_outputs, targets)  # shows each end use and how they change absolutely over the generations
        #plot_best_fit(cols_outputs, best_inds_fitness)

    plt.show()
    #TODO need to rearrange Y_real data so that days are consecutive instead of end-uses, will this change the prediction results???

if __name__ == '__main__':
    def start():
        print('start')
    
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    BuildingList = ['05_MaletPlaceEngineering', '01_CentralHouse', '02_BuroHappold_17', '03_BuroHappold_71']  # Location of DATA
    BuildingHardDisk = ['05_MaletPlaceEngineering_Project', '01_CentralHouse_Project', '02_BuroHappold_17', '03_BuroHappold_71']
    DataFilePaths = ['MPEB', 'CentralHouse', '17', '71']  # Location of STM data and file naming
    BuildingLabels = ['MPEB', 'Central House', 'Office 17', 'Office 71']
    BuildingAbbreviations = ['MPEB', 'CH', '17', '71', 'Nothing']
    InputVariables = ['inputs_MaletPlace11_08_16_11.csv', 'inputs_CentralHouse_22225_06_11_49.csv', '17', 'inputs_BH7102_09_14_04.csv']
    InputVariablesBaseCase = ['x', 'x', '17', 'inputs_basecase_BH71_05_09_11_39.csv']
    FloorAreas = [9579, 5876, 1924, 1691]

    building_num = 3 # 0 = MPEB, 1 = CH, 2 = 17, 3 = 71
    base_case = False # only show a single run from the basecase, or multiple runs (will change both import and plotting)
    NO_ITERATIONS = 100
    time_step = 'year'  # 'year', 'month', 'day', 'hour', 'half-hour' #todo only year and month work atm.
    end_uses = False # for month specifically (when True it will divide end-uses by month instead of aggregating to total)
    write_model = True # writes out the .pkl model.
    write_data = False # Writes out several .csv files some used for function optimisation script atm.
    plot_progression = False # this will plot the learning progression of the model when set to True.

    build_surrogate_model = False # run and build surrogate model, and sensitivity of real data
    feature_selection = False  # feature selection for when building a new model
    metamodel_sensitivity = False # use the built model for sobol
    function_optimisation = True # optimise towards measured data

    # OPTIMISATION INPUTS
    NGEN = 244  # is the number of generation for which the evolution runs  # For selTournamentDCD, pop size has to be multiple of four
    POPULATION_SIZE = 24  # no of individuals/samples
    CXPB = 0.9  # is the probability with which two individuals are crossed
    MUTPB = 0.5  # probability of mutating the offspring
    INDPB = 0.8  # Independent probability of each attribute to be mutated
    # PERC_ATTR_TO_MUTATE = 0.3 #no of attributes to mutate per individual




    building_abr = BuildingAbbreviations[building_num]
    datafile = DataFilePaths[building_num]
    building = BuildingList[building_num]
    building_harddisk = BuildingHardDisk[building_num]
    building_label = BuildingLabels[building_num]
    floor_area = FloorAreas[building_num]
    inputs = InputVariables[building_num]
    inputs_basecase = InputVariablesBaseCase[building_num]

    DataPath = 'C:/Users/' + getpass.getuser() + '/Dropbox/01 - EngD/07 - UCL Study/'
    DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/' + building_harddisk + '/ParallelSimulation/'
    DataPath_model_basecase = 'C:/EngD_hardrive/UCL_DemandLogic/' + building_harddisk + '/BaseCase/'



    if function_optimisation is True:


        # READ DATA
        if building_num in {0, 1}:  # Does it have short term monitoring?
            df_stm = ReadSTM(building, building_num, write_data, datafile, floor_area)  # read short term monitoring
        else:
            df_stm = pd.DataFrame()

        if building_num in {1}:  # does it have separate gas use?
            df_gas = ReadGas(building, building_num, write_data, datafile, floor_area)  # read gas data for Central House
        else:
            df_gas = pd.DataFrame()
        df, df_LVL1, df_floorsLP, df_mech, df_stm = ReadSubMetering(DataPath, building, building_num, building_abr, write_data, datafile, df_stm, floor_area)


        ### FUNCTION OPTIMISATION
        if time_step == 'year':
            if building_abr in {'71'}:
                df = df.loc['01-01-14 0:00':'31-12-14 23:30']
                df_LVL1 = df_LVL1.loc['01-01-14 0:00':'31-12-14 23:30']
            df_LVL1 = df_LVL1.sum(axis=0)
            print(df_LVL1)
        elif time_step == 'month':
            if end_uses is True:
                df_LVL1 = df_LVL1.resample('M').sum()
            else:
                df_LVL1 = df_LVL1.sum(axis=1).resample('M').sum()

        targets = df_LVL1
        targets = [int(i) for i in targets]


    # plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams.update({'font.size': 8})


    not_run = []
    run_numbers = []
    for i in range(NO_ITERATIONS):
        file_name = 'eplusmtr' + str(format(i, '04')) + '.csv'
        if os.path.isfile(os.path.join(DataPath_model_real, file_name)):
            run_numbers.append(i)
        else:
            not_run.append(str(format(i, '04')))
    main()

# TODO sensitivity analysis / sobol indices / uncertainty decomposition
# TODO for GPR, what kernel to use?
# TODO MARS not implemented in sklearn it seems, could use pyearth https://statcompute.wordpress.com/2015/12/11/multivariate-adaptive-regression-splines-with-python/
# TODO http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
# TODO have to cross-validate SVR the choice of parameters to increase regression accuracy

# TODO timeseries / monthly data.
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
