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
from collections import Sequence
from itertools import repeat
from pandas.plotting import parallel_coordinates
#import scipy
from scipy.optimize import curve_fit, leastsq, least_squares, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, sem
import scipy.stats as stats
from scipy.stats.mstats import zscore
from scipy.stats.kde import gaussian_kde
from sklearn.feature_selection import f_regression, mutual_info_regression
import matplotlib.ticker as mtick
#import statsmodel for VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

#Sensitivity modules
import SALib

from SALib.sample import morris, saltelli
from SALib.analyze import morris, sobol
from SALib.test_functions import Ishigami

#Sci-kit modules
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, TheilSenRegressor, HuberRegressor, RANSACRegressor # Linear Regression / Ordinary Least Squares
from sklearn.cross_decomposition import PLSRegression # Partial Least Squares Regression
from sklearn.svm import SVR, LinearSVR #Support Vector Regression
from sklearn.kernel_ridge import KernelRidge # Kernel Ridge Regression
from sklearn.gaussian_process import GaussianProcessRegressor # Kriging/GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C # GPR kernels
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
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
from PlotSubmetering import ReadRuns, ReadSTM, ReadGas
from PlotSubmetering import lineno
from PlotSubmetering import ReadSubMetering
from PlotSubmetering import ScaleDataFrame
from PlotSubmetering import CreateAvgProfiles


def PlotSurrogateLearningRate(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, ):
    its = np.arange(50, len(Y_train), 50)
    print(its)

    df_r2_combined = pd.DataFrame()
    df_mse = pd.DataFrame()
    df_scores_combined = pd.DataFrame()
    for i in range(1, len(Y_train)):
        if i % 50 == 0:
            print(i)
            mse_list, df_r2, df_scores, Y_test, prediction = SurrogateModel(X_train[:i], X_test, Y_train[:i], Y_test, cols_outputs, cols_inputs,
                                                time_step, end_uses, for_sensitivity, plot_progression=True, write_model=False, write_data=False, )


            #df_scores_combined = pd.concat([df_scores_combined, df_scores])
            df_r2_combined = pd.concat([df_r2_combined, df_r2], axis=1)
            df_mse = pd.concat([df_mse, pd.DataFrame(mse_list)], axis=1)

    if NO_ITERATIONS > 50:

        df_r2_combined.columns = its
        df_r2_combined = df_r2_combined.astype(float)
        df_mse.columns = its
        df_mse = df_mse.astype(float)

        # df_scores_combined.index = its
        # df_scores_combined.columns = ['MAE']
        # df_scores_combined = df_scores_combined.astype(float)

        ax = df_mse.T.plot(color=COLORS, figsize=(14 / 2.54, 6 / 2.54))
        #df_scores_combined.plot(ax=ax, color=COLORS[1])
        ax2 = ax.twinx()
        df_r2_combined.T.plot(ax=ax2, color=COLORS, linestyle='--')

    ax.set_ylabel('MAE [kWh]')
    ax2.set_ylabel('$\mathregular{r^{2}}$')

    ax2.set_ylim(0, 1)
    ax.set_ylim(0,None)
    ax2.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax2.set_axisbelow(True)

    ax.legend(bbox_to_anchor=(1.1,.8), loc='center left')
    ax2.legend(bbox_to_anchor=(1.1,.2), loc='center left')

    #plt.tight_layout(rect=[0, 0, 0.9, 1], pad=.95, w_pad=0.05, h_pad=0.05)  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_SurrogateLearningRate.png', dpi=300, bbox_inches='tight')

def PlotCompareElementaryEffects(building_label):
    df_morris_mu = pd.read_csv(DataPath_model_real + 'df_morris_mu.csv', index_col=0, header=0)
    df_morris_sigma = pd.read_csv(DataPath_model_real + 'df_morris_sigma.csv', index_col=0, header=0)

    # corrs = pd.concat([df_corr_spearman.iloc[:20, 1], df_corr_pearson.iloc[:20, 1], df_sobol_first.iloc[:20, 1], df_sobol_total.iloc[:20, 1]], axis=1)
    #corrs = pd.concat([df_morris_mu.iloc[:, 1], df_morris_sigma.iloc[:20, 1]], axis=1)

    fig = plt.figure(figsize=(14 / 2.54, 10 / 2.54))
    ax = fig.add_subplot(111)

    min_sigma, max_sigma = df_morris_sigma.min().min(), df_morris_sigma.max().max()
    min_mu, max_mu = 0, df_morris_mu.max().max()

    for i, col in enumerate(df_morris_sigma.columns):
        ax.scatter(df_morris_mu[col], df_morris_sigma[col], alpha=0.4, label=col, edgecolor=COLORS[i], facecolor='none')
        for j,v in enumerate(df_morris_mu[col]):
            print(df_morris_mu[col][j])
            if df_morris_mu[col][j] > (max_mu / 2.2):  #or df_morris_sigma[col][j] > (max_sigma / 2)
                ax.annotate(df_morris_mu.index[j], df_morris_mu[col][j], df_morris_sigma[col][j])

    print(min_sigma, max_sigma)
    print(min_mu, max_mu)
    #ax.set_title('Gas Predicted and Measured vs. Outdoor temperature, $R^2 = %0.2f$' % Rsqr, fontsize=8)

    ax.legend(loc='center left', bbox_to_anchor=(1.0, .5))
    ax.set_ylim(min_sigma, max_sigma)
    ax.set_xlim(min_mu, max_mu)
    ax.set_xlabel('Mu')
    ax.set_ylabel('Sigma')


    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    plt.savefig(DataPathImages + building_label + '_MorrisMethod.png', dpi=300, bbox_inches='tight')


def PlotCompareCoefficients(building_label):
    df_corr_standardized = pd.read_csv(DataPath_model_real + 'Correlations_METAMODELStandardizedyear.csv', index_col=0, header=0)
    df_corr_spearman = pd.read_csv(DataPath_model_real + 'Correlations_METAMODELSpearmanyear.csv', index_col=0, header=0)
    df_corr_pearson = pd.read_csv(DataPath_model_real + 'Correlations_METAMODELPearsonyear.csv', index_col=0, header=0)
    df_sobol_first = pd.read_csv(DataPath_model_real + 'df_sobol_first.csv', index_col=0, header=0)
    df_sobol_total = pd.read_csv(DataPath_model_real + 'df_sobol_total.csv', index_col=0, header=0)
    # df_morris_mu = pd.read_csv(DataPath_model_real + 'df_morris_mu.csv', index_col=0, header=0)
    # df_morris_sigma = pd.read_csv(DataPath_model_real + 'df_morris_sigma.csv', index_col=0, header=0)

    corrs = pd.concat([df_corr_spearman.iloc[:20, 1], df_corr_pearson.iloc[:20, 1], df_sobol_first.iloc[:20, 1], df_sobol_total.iloc[:20, 1]], axis=1)
    #corrs = pd.concat([df_morris_mu.iloc[:20, 1], df_morris_sigma.iloc[:20, 1]], axis=1)

    ax = corrs.plot(kind='bar', width=.8, color=COLORS, figsize=(14 / 2.54, 5 / 2.54))
    ax.legend(['Spearman','Pearson', 'Sobol_first', 'Sobol_total'], loc='upper left')

    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    ax.set_ylabel('Coefficient')
    ax.set_xlabel('')

    plt.savefig(DataPathImages + building_label + '_CompareCoefficients.png', dpi=300, bbox_inches='tight')

def PlotSurrogateStandardError(X_real, Y_real, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, ):
    df = pd.DataFrame(Y_real, columns=cols_outputs)
    df = df/(3600*1000) #kWh
    df_combined = pd.DataFrame()
    for i in range(1, len(df)):
        if i % 20 == 0:

            df_se = df[:i].std()/len(df[:i])
            df_se = pd.DataFrame(df_se, columns=[i])
            df_combined = pd.concat([df_combined, df_se], axis=1)

    ax = df_combined.T.plot(figsize=(14 / 2.54, 5 / 2.54))
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    ax.set_ylabel('Standard error [kWh]')
    ax.set_xlabel('Number of simulations')
    ax.legend(bbox_to_anchor=(1,.5), loc='center left')

    #plt.tight_layout(rect=[0, 0, 0.9, 1], pad=.95, w_pad=0.05, h_pad=0.05)  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_StandardErrorSurrogate.png', dpi=300, bbox_inches='tight')


def PlotSurrogateSpearmanConvergence(X_real, Y_real, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, ):
    df_combined = pd.DataFrame()
    its = []
    for i in range(1, len(Y_real)):
        if i % 20 == 0:
            its.append(i)
            df_corr = Correlations(DataPath_model_real, X_real[:i], Y_real[:i], cols_outputs, cols_inputs, time_step, metamodel_sensitivity)
            df_combined = pd.concat([df_combined, df_corr.iloc[:10,1]], axis=1)

    df_combined.columns=its
    ax = df_combined.T.plot(figsize=(14 / 2.54, 5 / 2.54))
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Number of simulations')
    ax.legend(bbox_to_anchor=(1, .5), loc='center left')

    # plt.tight_layout(rect=[0, 0, 0.9, 1], pad=.95, w_pad=0.05, h_pad=0.05)  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_SpearmanConvergence.png', dpi=300, bbox_inches='tight')

def PlotSurrogateSobolConvergence(X_real, Y_real, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, ):
    df_combined = pd.DataFrame()
    its = []


    norm_bounds = pd.concat([pd.DataFrame(mus), pd.DataFrame(sigmas)], axis=1)
    norm_bounds = norm_bounds.values.tolist()  # turn into list [[mu, sigma], [mu, sigma] ...]
    D = len(cols_inputs)  # no. of parameters
    N = 1000 # size of an initial monte carlo sample, 2000 simulations?
    second_order = True
    if second_order == True:
        samples = N * (2 * D + 2)
    else:
        samples = N * (D + 2)

    real_cols = cols_inputs  # columns names of the parameters
    problem = {'num_vars': D, 'names': real_cols, 'bounds': norm_bounds * D}  # bound set to the input parameters

    #X_real, Y_real = PKL_SurrogateModel(samples, mus, sigmas, lower_limits, upper_limits)

    df_inputs = pd.DataFrame(X_real, columns=[cols_inputs])
    df_outputs = pd.DataFrame(Y_real, columns=[cols_outputs])
    df_outputs = df_outputs / (3600*1000) / floor_area

    ## Analyze Sobol
    df_sobol_first = pd.DataFrame()
    df_sobol_total = pd.DataFrame()

    for i in range(1, len(Y_real)):
        if i % 20 == 0:
            its.append(i)
            df_corr = Correlations(DataPath_model_real, X_real[:i], Y_real[:i], cols_outputs, cols_inputs, time_step, metamodel_sensitivity)
            df_combined = pd.concat([df_combined, df_corr.iloc[:10,1]], axis=1)


            for i,v in enumerate(df_outputs.iloc[:,:].columns):
                Si_sobol = sobol.analyze(problem, df_outputs[v].values, calc_second_order=second_order)
                df_first = pd.DataFrame(Si_sobol['S1'].tolist(), index=cols_inputs, columns=[str(v)])
                df_total = pd.DataFrame(Si_sobol['ST'].tolist(), index=cols_inputs, columns=[str(v)])
                df_sobol_first = pd.concat([df_sobol_first, df_first], axis=1)
                df_sobol_total = pd.concat([df_sobol_total, df_total], axis=1)



    df_combined.columns=its
    ax = df_combined.T.plot(figsize=(14 / 2.54, 5 / 2.54))
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Number of simulations')
    ax.legend(bbox_to_anchor=(1, .5), loc='center left')

    # plt.tight_layout(rect=[0, 0, 0.9, 1], pad=.95, w_pad=0.05, h_pad=0.05)  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_SpearmanConvergence.png', dpi=300, bbox_inches='tight')

def PlotSobolIndices(Si_sobol, cols_outputs, cols_inputs,):
    D = len(cols_inputs)
    bar_width = .45

    fig = plt.figure(0)
    ax = plt.subplot(111)
    y_pos = np.arange(D)
    # print(y_pos)

    df_first = pd.DataFrame(Si_sobol['S1'].tolist(), index=cols_inputs)
    df_total = pd.Series(Si_sobol['ST'].tolist(), index=cols_inputs)

    print(df_first)
    ax.plot(df_first.iloc[:,0], label='First-order')
    #ax.plot(df_total, label='Total order')

    print(cols_inputs[0], '-', cols_inputs[1], Si_sobol['S2'][0, 1])

    # plt.xticks(y_pos + bar_width / 2, real_cols, rotation='vertical')
    # plt.tight_layout()
    plt.legend()

def PlotSurrogateModelPerformance(df_scores, df_r2, building_label):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(16 / 2.54, 6 / 2.54))  # width, height
    ax = axes[0]
    ax2 = axes[1]

    df_scores.plot(ax=ax, kind='bar', stacked=False, width=.8, position=0.5)
    df_r2.plot(ax=ax2, kind='bar', width=.8, position=0.5, color=COLORS[5])

    print(df_scores)
    print(df_r2)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])

    # ax.legend(bbox_to_anchor=(0,1.1), loc='center right')
    # ax2.legend(bbox_to_anchor=(1,1.1), loc='center left')
    ax.legend_.remove()
    ax2.legend_.remove()
    ax.set_ylabel('MAE [kWh]')
    ax2.set_ylabel('R2')

    ax2.set_ylim(0, 1)
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    ax2.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_MetaModelPerformance.png', dpi=300, bbox_inches='tight')

    # if plot_progression == False:
    #     fig = plt.figure()
    #     ax1 = plt.subplot(111)
    #     for i in range(len(cols_outputs[:9])):
    #         prediction_series = pd.Series(prediction[:, i], name=cols_outputs[i] + '_' + name)
    #         predictions = pd.concat([predictions, prediction_series], axis=1)
    #
    #         prediction_std = np.std(predictions)
    #         # print(cols_outputs[i], name, prediction[:5], Y_test[:5])
    #         ax1.plot(prediction[:, i], color=COLORS[x], label=name if i == 0 else "")  # plot first column, and label once

def PlotValidateModelFit(Y_test, prediction, cols_outputs, time_step):
    df_y_std = pd.DataFrame(Y_test, columns=cols_outputs).std()
    df_y_mean = pd.DataFrame(Y_test, columns=cols_outputs).mean()
    df_p_std = pd.DataFrame(prediction, columns=cols_outputs).std()
    df_p_mean = pd.DataFrame(prediction, columns=cols_outputs).mean()

    #df_result = abs(df_y.subtract(df_p))

    if time_step == 'month':
        fig, axes = plt.subplots(nrows=3, ncols=1, sharey=False, figsize=(7 / 2.54, 8 / 2.54))

        ax = df_y_mean[:12].plot(ax=axes[0], color=COLORS[0])
        df_p_mean[:12].plot(ax=axes[0], color=COLORS[1])
        ax.fill_between(np.arange(0, 12), df_y_mean[:12]-df_y_std[:12], df_y_mean[:12]+df_y_std[:12], color=COLORS[0], alpha=.2)
        ax.fill_between(np.arange(0, 12), df_p_mean[:12] - df_p_std[:12], df_p_mean[:12] + df_p_std[:12], color=COLORS[1], alpha=.2)

        ax2 = df_y_mean[12:24].plot(ax=axes[1], color=COLORS[0])
        df_p_mean[12:24].plot(ax=axes[1], color=COLORS[1])
        ax2.fill_between(np.arange(0, 12), df_y_mean[12:24]-df_y_std[12:24], df_y_mean[12:24]+df_y_std[12:24], color=COLORS[0], alpha=.2)
        ax2.fill_between(np.arange(0, 12), df_p_mean[12:24] - df_p_std[12:24], df_p_mean[12:24] + df_p_std[12:24], color=COLORS[1], alpha=.2)

        ax3 = df_y_mean[24:36].plot(ax=axes[2], color=COLORS[0])
        df_p_mean[24:36].plot(ax=axes[2], color=COLORS[1])
        ax3.fill_between(np.arange(0, 12), df_y_mean[24:36] - df_y_std[24:36], df_y_mean[24:36] + df_y_std[24:36], color=COLORS[0], alpha=.2)
        ax3.fill_between(np.arange(0, 12), df_p_mean[24:36] - df_p_std[24:36], df_p_mean[24:36] + df_p_std[24:36], color=COLORS[1], alpha=.2)

        ax.set_ylim(0,None)
        ax2.set_ylim(0, None)
        ax3.set_ylim(0, None)

        ticklabel_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        ax.set_ylabel('Lights')
        ax2.set_ylabel('Power')
        ax3.set_ylabel('Gas')

        ticklabels = ['Simulations', 'Meta-model']
        ax.legend(ticklabels, bbox_to_anchor=(1.0,.5))

        ind_x = np.arange(df_y_mean[:12].shape[0])
        ax3.set_xticks(ind_x)  # set positions for y-labels, .5 to put the labels in the middle

        ax3.set_xticklabels(ticklabel_months, rotation=90)
        xticks = ax.xaxis.get_major_ticks()
        xticks2 = ax2.xaxis.get_major_ticks()
        for index, label in enumerate(ax.get_xaxis().get_ticklabels()):
            xticks[index].set_visible(False) # hide ticks where labels are hidden
        for index, label in enumerate(ax2.get_xaxis().get_ticklabels()):
            xticks2[index].set_visible(False) # hide ticks where labels are hidden

        fig.subplots_adjust(hspace=.1)  # space between plots
    else:

        ax = df_result.plot(color=COLORS[:12], figsize=(16 / 2.54, 8 / 2.54))
        #df_target.iloc[:, :].plot(ax=ax, style='--', color=COLORS[:12])

        ax.set_ylim(0,None)
        ax.set_title('Prediction of the best individual per generation', fontsize=9)
        ax.set_xlabel('Generations')
        ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
        ax.set_axisbelow(True)

        # handles, labels = ax.get_legend_handles_labels()
        # print(labels)
        # print(len(labels)/2)
        # labels, handles = labels[int(len(handles)/2):], handles[int(len(handles)/2):] #reverse labels!

        ax.legend(loc='center left', bbox_to_anchor=(1.0,.5), title='Objectives')
        ax.set_ylabel('Energy $\mathregular{(kWh/m^{2}a)}$')
        #plt.tight_layout()

    plt.savefig(DataPathImages + building_label + '_' + time_step + '_PlotMetaModelFit.png', dpi=300, bbox_inches='tight')

def PlotParallelCoordinates(df_inputs, df_computed_inputs, df_outputs, building_label):
    #create output dataframe to scale on, these are the predictions
    df_outputs_stats = pd.concat([pd.DataFrame(df_outputs.mean()).T, pd.DataFrame(df_outputs.mean()*.2).T,
                                  pd.DataFrame(df_outputs.min()).T, pd.DataFrame(df_outputs.max()).T,
                                  pd.DataFrame([.2 for i in range(df_outputs.shape[1])], index=df_outputs.columns.tolist()).T], axis=0)
    df_outputs_stats = df_outputs_stats.reset_index()
    df_inputs = pd.concat([df_inputs, df_outputs_stats], axis=1) #combine the stats from inputs (mean, std etc with those from outputs)
    df = pd.concat([df_computed_inputs, df_outputs], axis=1)

    fig = plt.figure(figsize=(16 / 2.54, 23/ 2.54))
    ax = fig.add_subplot(111)

    df = df.loc[:, (df != 0).any(axis=0)]
    cols_included = df.columns.tolist()
    df_inputs = df_inputs[cols_included]

    print(df_inputs.shape, df.shape)

    scaled_frames = []
    for i, v in enumerate(df.columns):
        scaled_frames.append((df[v]-df[v].min())/(df[v].max()-df[v].min()))
    df = pd.concat(scaled_frames, axis=1)


    for i,v in enumerate(df.index):

        #if df['Gas'].loc[v] < .3 and df['Equipment'].loc[v] < .3:
        if df['HW Boiler 1'].loc[v] > .7 and df['HW Boiler 2'].loc[v] > .7:
            ax.plot(df.loc[v], range(1,df.shape[1]+1), c=COLORS[1], mec='k', ms=3, marker="o", lw=1, zorder=10, alpha=.8, linestyle='--')
        else:
            ax.plot(df.loc[v], range(1, df.shape[1] + 1), c='gray', lw=1, zorder=5, alpha=.3, linestyle='-')

    ax.set_ylim(1, df.shape[1])
    yticks = range(1,df.shape[1]+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(df.columns.tolist(), fontsize=5)
    ax.set_xlim(0,1)
    ranges = ['[' + str(i) + ', ' + str(j) + ']' for i, j in zip(df_inputs[cols_included].iloc[2], df_inputs[cols_included].iloc[3])]
    ax2 = ax.twinx()
    ax2.set_ylim(1, df.shape[1])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ranges, fontsize=5)

    plt.tight_layout()  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_ParallelCoordinates.png', dpi=300, bbox_inches='tight')

## PLOT Correlations ##
def PlotCorrelations(df_corr, cols_outputs):
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
def HeatmapCorrelations(df, df_outputs, cols_outputs, include_sobol, building_label):
    df_perc_total = [i / df_outputs.mean(axis=0).sum() for i in df_outputs.mean(axis=0)]
    print(df_perc_total)
    df_standard = df.multiply(df_perc_total)
    print(df_standard.mean().mean(), df_standard.max().max(), df_standard.min().min())
    df_standard = (df_standard-df_standard.mean().mean()) / (df_standard.max().max() - df_standard.min().min())

    #df = df[df[df > 10].any(1)] # select rows where any column is greater than 10
    #df = df[abs(df[abs(df) > .25].count(axis=1) > 0.35)] # where any value is greater than .5 and any column is greater than 1

    df_index = df[abs(df[abs(df) > .25].count(axis=1) > 0.25)]
    df_index_standard = df_standard[abs(df_standard[abs(df_standard) > .25].count(axis=1) > 0.25)]
    print(df_index.index.tolist())

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    df_index = f7(df_index.index.tolist()+df_index_standard.index.tolist())
    #df.filter(like=df_index, axis=0)
    df = df.loc[df_index]
    df_standard = df_standard.loc[df_index]

    # print(df)
    # print(df.shape)
    #df = df[abs(df['Cooling'] > .25)]
    #df = df[abs(df.iloc[:,:] > .1)]

    #vif_indices = [i for i, v in enumerate(vif) if v == np.inf or v < 6]  # when to exclude features
    dfs = [df_standard, df]
    cols_outputs_add = [v+' ['+str(round(df_perc_total[i]*100,1))+'%]' for i,v in enumerate(cols_outputs)]
    for i in range(len(dfs)):
        df = dfs[i]
        heatmap = df.as_matrix(columns=cols_outputs)
        fig, ax = plt.subplots(figsize=(18/ 2.54, 16 / 2.54))

        use_sns = True
        if use_sns is True:
            #sns.set(font_scale=0.9)
            ax = sns.heatmap(heatmap, linewidths=.8, annot=True,  cmap='RdBu_r', annot_kws={"size": 6}, fmt='.2f', vmin=-1, vmax=1) #cmap=cm.Spectral_r,
            ax.set_aspect("equal")
            ax.set_yticklabels(df.index, rotation=0)  # set y labels ('variables') from index
            ax.set_xticklabels(cols_outputs_add, rotation=90)  # set y labels ('variables') from index
            ax.set_aspect('equal')
            ax.xaxis.tick_top()

        else:
            im = ax.matshow(heatmap, cmap=cm.Spectral_r, fmt='d', interpolation='none')
            cbar = plt.colorbar(im, fraction=0.04555, pad=0.04)
            cbar.ax.tick_params()

            ind_x = np.arange(df.shape[1])
            ind_y = np.arange(df.shape[0])
            print(ind_x, ind_y)

            ax.set_aspect('equal')
            ax.set_yticks(ind_y)  # set positions for y-labels, .5 to put the labels in the middle
            ax.set_yticklabels(df.index, rotation = 0)  # set y labels ('variables') from index
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

        #ax.legend(bbox_to_anchor=(1, .5), loc='center left')  # bbox_to_anchor=(all the way next to the plot =1 if center left = loc, height 0.5 is in the middle)
        #plt.tight_layout(rect=[0, 0, 0.8, 1], pad=.75, w_pad=0.1, h_pad=0.1)  # use rect to adjust the plot sides rects=[left, bottom, right, top]
        if include_sobol is True:
            sobol = '_sobol'
        else:
            sobol=''


        plt.savefig(DataPathImages + building_label + sobol + str(i) + '_HeatMapCorrelations.png', dpi=400, bbox_inches='tight')

def ScatterCorrelation(df_input, df_output,building_label, var_i, var_o, ):
    #todo potentially allow for plotting multiple outputs, inputs wlil be difficult with different values.
    input_col = df_input.columns.tolist()
    output_col = df_output.columns.tolist()

    input = df_input[var_i]
    output = df_output[var_o]

    fig = plt.figure(figsize=(6/ 2.54, 6 / 2.54))
    ax = fig.add_subplot(111)

    reorder = sorted(range(len(input)), key = lambda ii: input[ii])
    xd = [input[ii] for ii in reorder]
    yd = [output[ii] for ii in reorder]
    par = np.polyfit(xd, yd, 1, full=True)

    slope=par[0][0]
    intercept=par[0][1]
    xl = [min(xd), max(xd)]
    yl = [slope*xx + intercept  for xx in xl]

    # coefficient of determination, plot text
    variance = np.var(yd)
    residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
    Rsqr = np.round(1-residuals/variance, decimals=2)

    # error bounds
    yerr = [abs(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)]
    par = np.polyfit(xd, yerr, 2, full=True)

    yerrUpper = [(xx*slope+intercept)+(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(xd,yd)]
    yerrLower = [(xx*slope+intercept)-(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(xd,yd)]

    ax.plot(xl, yl, '-', color=COLORS[1])
    ax.plot(xd, yerrLower, '--', color=COLORS[1])
    ax.plot(xd, yerrUpper, '--', color=COLORS[1])

    max_dots = 500
    ax.scatter(df_input[var_i][:max_dots], df_output[var_o][:max_dots], alpha=.8)
    #ax.plot(x, m*x + b, '-')
    #ax.set_xlim(0, ),

    ax.set_xlabel(var_i)
    ax.set_ylabel(var_o + ' energy use [kWh]')
    ax.set_title('$R^2 = %0.2f$'% Rsqr, fontsize=9)
    plt.savefig(DataPathImages + building_label + '_ScatterSingleVariable.png', dpi=300, bbox_inches='tight',
                figsize=(16 / 2.54, 8 / 2.54))

# PLOT: Plot predictions based on best individuals in each generation
def PlotPredictionGen():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    gen = logbook.select("gen")
    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i], for_sensitivity)
        result.append(prediction)
        #print('best indices used array:', best_inds[i])
        #print('best indices used for prediction', prediction)

    for i,v in enumerate(result):
        alp = 1-(i/50)
        ax.plot(v, label=gen[i], alpha=alp)
    ax.plot(targets, 'o', color='black', label='Target')
    ax.legend(loc='best')
    ax.set_title('ax2, best individual per generation',fontsize=9)




def PlotUncertainty(df_outputs, df_inputs, cols_outputs, time_step, end_uses, building_abr, building_label):
    if building_abr in {'71'}:
        no_months = 12
        ticklabel_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        cols_new = [i.split(' ', 1)[0] for i in cols_outputs[::no_months]]
    else:
        no_months = 8

    fig = plt.figure(figsize=(16 / 2.54, 5 / 2.54))
    ax = fig.add_subplot(111)

    if time_step == 'year' and end_uses is True:
        df_outputs_perc = (df_outputs.std() * 100) / df_outputs.mean()
        df_outputs_std = df_outputs.std()

        ax = df_outputs_std.plot(kind='bar', width=.4, color=COLORS[0], edgecolor='black', position=0)
        ax2 = ax.twinx()
        df_outputs_perc.plot(ax=ax2, kind='bar', width=.4, color=COLORS[1], edgecolor='black', position=1)

        ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
        ax.set_axisbelow(True)

        ax2.set_ylabel('% of mean')
        ax.set_ylabel('kWh')
        plt.title('Standard deviation in annual energy use', fontsize=9)

        plt.savefig(DataPathImages + building_label + '_' + time_step + '_Uncertainty.png', dpi=300, bbox_inches='tight')

    if time_step == 'month' and end_uses is True:

        print('h')

        # df_outputs_perc = (df_outputs.std() * 100) / df_outputs.mean()
        # df_outputs_std = pd.DataFrame(df_outputs.std().values.reshape(-1, len(ticklabel_months)), columns=ticklabel_months, index=cols_new).T
        # df_outputs_perc = pd.DataFrame(df_outputs_perc.values.reshape(-1, len(ticklabel_months)), columns=ticklabel_months, index=cols_new).T
        # print(df_outputs_std)
        #
        # fig, axes = plt.subplots(nrows=len(cols_new), ncols=1, sharey=False)
        #
        # for i, v in enumerate(cols_new):
        #     ax = df_outputs_std[v].plot(ax=axes[i], x=range(len(ticklabel_months)), kind='bar', width=.4, color=COLORS[0], edgecolor='black', position=0)
        #     ax2 = ax.twinx()
        #     df_outputs_perc[v].plot(ax=ax2, kind='bar', width=.4, color=COLORS[1], edgecolor='black', position=1)
        #     ax.set_ylim(0, None)
        #     ax.set_ylabel(v, rotation=0)
        #
        # fig.subplots_adjust(hspace=.1)  # space between plots

# PLOT: Plot energy use for the objectives with increasing generation
def PlotObjectiveConvergence(best_inds, cols_outputs, targets, floor_area, time_step, end_uses, building_label):
    if end_uses is True:
        end_use = '_end_uses'
    else:
        end_use = ''

    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i], for_sensitivity)
        result.append(prediction)

    df_result = pd.DataFrame(result, columns=cols_outputs)

    df_result = df_result / floor_area
    targets_dup = ([targets, ]) * len(result)  # duplicate targets list

    df_target = pd.DataFrame(targets_dup, columns=cols_outputs)
    df_target = df_target / floor_area

    df_result = abs(df_result.subtract(df_target))
    # df_target = pd.DataFrame(targets_dup, columns=cols_outputs)

    if time_step == 'month' and end_uses is True:
        fig, axes = plt.subplots(nrows=3, ncols=1, sharey=False)

        # todo this won't work for additional end-uses...
        ax = df_result[cols_outputs[:12]].plot(ax=axes[0], color=COLORS[:12])
        ax2 = df_result[cols_outputs[12:24]].plot(ax=axes[1], color=COLORS[:12])
        ax3 = df_result[cols_outputs[24:36]].plot(ax=axes[2], color=COLORS[:12])
        ax.legend(loc='center left', bbox_to_anchor=(1.0, .5))
        ax2.legend(loc='center left', bbox_to_anchor=(1.0, .5))
        ax3.legend(loc='center left', bbox_to_anchor=(1.0, .5))
        ax.set_ylim(0,None)
        ax2.set_ylim(0, None)
        ax3.set_ylim(0, None)

        ax3.set_xlabel('Generations')
        ax.set_ylabel('Lights $\mathregular{(kWh/m^{2}a)}$')
        ax2.set_ylabel('Power $\mathregular{(kWh/m^{2}a)}$')
        ax3.set_ylabel('Gas $\mathregular{(kWh/m^{2}a)}$')
        ax2.legend_.remove()
        ax3.legend_.remove()
        ticklabel_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.legend(ticklabel_months, bbox_to_anchor=(1.0,.5))

        xticks = ax.xaxis.get_major_ticks()
        xticks2 = ax2.xaxis.get_major_ticks()
        for index, label in enumerate(ax.get_xaxis().get_ticklabels()):
            xticks[index].set_visible(False) # hide ticks where labels are hidden
        for index, label in enumerate(ax2.get_xaxis().get_ticklabels()):
            xticks2[index].set_visible(False) # hide ticks where labels are hidden

        fig.subplots_adjust(hspace=.1)  # space between plots
    else:
        ax = df_result.plot(color=COLORS[:12], figsize=(16 / 2.54, 6 / 2.54))
        #df_target.iloc[:, :].plot(ax=ax, style='--', color=COLORS[:12])
        ax.set_ylim(0,None)
        ax.set_title('Prediction of the best individual per generation', fontsize=9)
        ax.set_xlabel('Generations')
        ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
        ax.set_axisbelow(True)

        # handles, labels = ax.get_legend_handles_labels()
        # print(labels)
        # print(len(labels)/2)
        # labels, handles = labels[int(len(handles)/2):], handles[int(len(handles)/2):] #reverse labels!

        ax.legend(loc='center left', bbox_to_anchor=(1.0,.5), title='Objectives')
        ax.set_ylabel('Energy $\mathregular{(kWh/m^{2}a)}$')
        #plt.tight_layout()
    plt.savefig(DataPathImages + building_label + '_' + time_step + end_use + weekday_str + '_ObjectiveConvergence.png', dpi=300, bbox_inches='tight')


# PLOT: Plot energy use for the objectives with increasing generation
def PlotObjectiveTimeConvergence(best_inds, cols_outputs, targets, floor_area, time_step, end_uses, building_label):
    if end_uses is True:
        end_use = '_end_uses'
    else:
        end_use = ''

    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i], for_sensitivity)
        result.append(prediction)

    df_result = pd.DataFrame(result, columns=cols_outputs)

    df_result = df_result / floor_area
    targets_dup = ([targets, ]) * len(result)  # duplicate targets list

    df_target = pd.DataFrame(targets_dup, columns=cols_outputs)
    df_target = df_target / floor_area

    df_result = abs(df_result.subtract(df_target))
    # df_target = pd.DataFrame(targets_dup, columns=cols_outputs)

    if time_step == 'month' and end_uses is True:
        fig, axes = plt.subplots(nrows=3, ncols=1, sharey=False)

        # todo this won't work for additional end-uses...

        ax = df_result[cols_outputs[36:60]].plot(ax=axes[0], color=COLORS[:12])
        ax2 = df_result[cols_outputs[60+24:60+24*2]].plot(ax=axes[1], color=COLORS[:12])
        ax3 = df_result[cols_outputs[60+24*3:60+24*4]].plot(ax=axes[2], color=COLORS[:12])
        ax.legend(loc='center left', bbox_to_anchor=(1.0, .5))
        ax2.legend(loc='center left', bbox_to_anchor=(1.0, .5))
        ax3.legend(loc='center left', bbox_to_anchor=(1.0, .5))
        ax.set_ylim(0,None)
        ax2.set_ylim(0, None)
        ax3.set_ylim(0, None)

        ax3.set_xlabel('Generations')
        ax.set_ylabel('Weekday (Lights) $\mathregular{(kWh/m^{2}a)}$')
        ax2.set_ylabel('Weekend (Power) $\mathregular{(kWh/m^{2}a)}$')
        ax3.set_ylabel('Weekday (Gas) $\mathregular{(kWh/m^{2}a)}$')
        ax2.legend_.remove()
        ax3.legend_.remove()
        ticklabel_months = [str(i) for i in range(24)]
        ax.legend(ticklabel_months, bbox_to_anchor=(1.0,.5))

        xticks = ax.xaxis.get_major_ticks()
        xticks2 = ax2.xaxis.get_major_ticks()
        for index, label in enumerate(ax.get_xaxis().get_ticklabels()):
            xticks[index].set_visible(False) # hide ticks where labels are hidden
        for index, label in enumerate(ax2.get_xaxis().get_ticklabels()):
            xticks2[index].set_visible(False) # hide ticks where labels are hidden

        fig.subplots_adjust(hspace=.1)  # space between plots
    else:
        ax = df_result.plot(color=COLORS[:12], figsize=(16 / 2.54, 6 / 2.54))
        #df_target.iloc[:, :].plot(ax=ax, style='--', color=COLORS[:12])
        ax.set_ylim(0,None)
        ax.set_title('Prediction of the best individual per generation', fontsize=9)
        ax.set_xlabel('Generations')
        ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
        ax.set_axisbelow(True)

        # handles, labels = ax.get_legend_handles_labels()
        # print(labels)
        # print(len(labels)/2)
        # labels, handles = labels[int(len(handles)/2):], handles[int(len(handles)/2):] #reverse labels!

        ax.legend(loc='center left', bbox_to_anchor=(1.0,.5), title='Objectives')
        ax.set_ylabel('Energy $\mathregular{(kWh/m^{2}a)}$')
        #plt.tight_layout()
    plt.savefig(DataPathImages + building_label + '_' + time_step + end_use + weekday_str + '_ObjectiveTimeConvergence.png', dpi=300, bbox_inches='tight')

# PLOT: Plot objective differences using RMSE
def PlotBestFit(cols_outputs, best_inds_fitness):
    #df_avg = pd.DataFrame(fit_avgs, columns=cols_objectives)
    #df_mins = pd.DataFrame(fit_mins, columns=cols_objectives)
    df_inds_fit = pd.DataFrame(best_inds_fitness, columns=cols_outputs)

    ax = df_inds_fit.plot(color=COLORS)
    ax.set_title("Best Individual fitnesses per gen", fontsize=9)

def PlotCalibratedSolutionsBoxplot(df_inputs, df_computed_inputs, df_calibrated, lower_limits, upper_limits, building_label):
    fig = plt.figure(figsize=(16 / 2.54, 23/ 2.54))
    ax = fig.add_subplot(111)

    df_computed_inputs = df_computed_inputs.loc[:, (df_computed_inputs != 0).any(axis=0)]
    cols_included = df_computed_inputs.columns.tolist()

    scaled_frames = []
    for i, v in enumerate(df_calibrated.columns):
        scaled_frames.append((df_calibrated[v]-lower_limits[i])/(upper_limits[i]-lower_limits[i]))

    df_calibrated = pd.concat(scaled_frames, axis=1)
    df_calibrated = df_calibrated[cols_included]

    bplot = df_calibrated[cols_included].plot.box(ax=ax, vert=False, showfliers=False, patch_artist=True, return_type='dict')  # showmeans=True
    ax.plot(df_calibrated[cols_included].loc[df_calibrated.shape[0]-1], list(range(1, df_calibrated[cols_included].shape[1]+1)), c=COLORS[1], mec='k', ms=5, marker="o", lw=1, zorder=10, alpha=.75, linestyle='None')

    # print(bplot.keys())
    # ax2.legend_.remove()
    for key in ['boxes', 'whiskers', 'caps']:
        for y, box in enumerate(bplot[key]): #set colour of boxes
                box.set(color=COLORS[0])
                box.set(linewidth=1)
    [i.set(color='black') for i in bplot['medians']]

    #ax.set_aspect(1.1)
    #fig.subplots_adjust(wspace=1)

    ax.set_ylim(1, df_calibrated.shape[1])
    yticks = range(1,df_calibrated.shape[1]+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(df_calibrated.columns.tolist(), fontsize=5)
    ranges = ['['+str(i)+', '+str(j)+']' for i, j in zip(df_inputs[cols_included].iloc[2], df_inputs[cols_included].iloc[3])]

    ax2 = ax.twinx()
    ax2.set_ylim(1, df_calibrated.shape[1])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ranges, fontsize=5)
    ax.set_xlim(0, 1)


    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.25)
    ax.set_axisbelow(True)
    #ax.xaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.25)
    #ax.set_axisbelow(True)
    #ax.set_xlabel(col)
    #ax.legend(labels, bbox_to_anchor=(0,1.1), loc='center left')

    plt.tight_layout()  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_CalibratedVariablesBoxplot.png', dpi=300, bbox_inches='tight')


def PlotCalibratedSolutions(df_inputs, df_calibrated, building_label):
    # let's plot the variation that can be plotted within and the variation of the individuals which are calibrated.
    #print(df_calibrated.head())
    fig = plt.figure(figsize=(8 / 2.54, 7/ 2.54))
    ax = fig.add_subplot(111)
    cols_inputs = df_calibrated.columns.tolist()


    labels = []

    for col in cols_inputs[3:4]:
        labels.append(col+str('_kde'))
        variable1 = df_calibrated[col].tolist()
        variable2 = df_inputs[col].tolist()
        variables = [variable1, variable2]
        for variable in variables:
            kde = gaussian_kde(variable)
            dist_space = linspace(min(variable), max(variable), 100)
            ax.plot(dist_space, kde(dist_space), label="kde")

    ax.set_label('COP')
    # for col in cols_inputs[3:4]:
    #     labels.append(col + str('_pdf'))
    #     variable1 = df_calibrated[col].tolist()
    #     variable2 = df_inputs[col].tolist()
    #     variables = [variable1, variable2]
    #     for variable in variables:
    #         variable.sort()
    #         print(variable)
    #         hmean = np.mean(variable)
    #         hstd = np.std(variable)
    #         pdf_variable = stats.norm.pdf(variable, hmean, hstd)
    #         ax.plot(variable, pdf_variable, label='pdf')

    ax.set_ylim(0, None)
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=.4)
    ax.set_axisbelow(True)
    ax.set_ylabel('Distribution')

    ax.legend(labels, bbox_to_anchor=(0,1.1), loc='center left')
    plt.tight_layout(rect=[0, 0, 1, 1], pad=.95, w_pad=0.05, h_pad=0.05)  # L D R U
    plt.savefig(DataPathImages + building_label + '_' + time_step + '_CalibratedVariables.png', dpi=300, bbox_inches='tight')

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

def Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step, metamodel_sensitivity):
    #todo how about correlations on total energy use?, like an extra bar
    print(lineno(), 'no. variables', len(cols_inputs))
    print(spearmanr(X_real[:,0], Y_real[:,0])[0])
    print(pearsonr(X_real[:,0], Y_real[:,0])[0])

    label = ['Standardized', 'Spearman', 'Pearson']

    for p, q in enumerate(label):
        df_corr = pd.DataFrame(cols_inputs)
        df_corr.columns = [q]
        for j in range(Y_real.shape[1]):
            coef_list = []
            for i in range(X_real.shape[1]):
                if p == 0:
                    coef_list.append(sm.OLS(zscore(X_real[:, i]), zscore(Y_real[:, j])).fit().params[0])
                elif p == 1:
                    coef_list.append(spearmanr(X_real[:, i], Y_real[:, j])[0])
                elif p == 2:
                    coef_list.append(pearsonr(X_real[:, i], Y_real[:, j])[0])
            df_corr[cols_outputs[j]] = pd.Series(coef_list) #append list to df
        df_corr.set_index(q, inplace=True)
        print(df_corr.head())

        if metamodel_sensitivity is True:
            df_corr.to_csv(DataPath_model_real + 'Correlations_METAMODEL' + q + time_step + '.csv')
        else:
            df_corr.to_csv(DataPath_model_real + 'Correlations' + q + time_step + '.csv')

        if p == 0:
            df_corr_standardized = df_corr
        elif p == 1:
            df_corr_spearman = df_corr
        elif p == 2:
            df_corr_pearson = df_corr

    return df_corr, df_corr_standardized, df_corr_spearman, df_corr_pearson

def ImportOutputs(DataPath_model_real, inputs, Y_real, cols_outputs, run_numbers, feature_selection):
    df = pd.read_csv(DataPath_model_real + inputs, header=0)
    cols_inputs = df.columns.tolist()
    df_in = df.copy()

    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)

    X_real = df.ix[run_numbers]
    X_real = X_real.drop(df.columns[[0]], axis=1) #drop run_no column
    X_real = X_real.as_matrix()


    # FEATURE SELECTION
    print(lineno(), len(cols_inputs), cols_inputs)
    if feature_selection is True:

        # VIF selection # multicollinearity, which should definitely not exist in the model keep values below 6
        VIF_selection = False
        if VIF_selection is True:
            X_vars = np.append(X_real, np.ones([len(X_real), 1]), 1) # need to add a column of ones at the end...
            vif = [variance_inflation_factor(X_vars, i) for i in range(X_vars.shape[1])] # Catch infs if not enough runs
            vif_indices = [i for i, v in enumerate(vif) if v == np.inf or v < 20] # when to exclude features
            vif_indices = vif_indices[:-1] ## need to remove the last item as it is an added zero value.
            vif = vif[:-1]

            print('max vif:', np.max(vif))
            print(lineno(), 'vif indices', vif_indices)
            df_vif = pd.DataFrame(vif, index=cols_inputs)
            df_vif.to_csv(DataPath_model_real+'vif.csv')
            print(df_vif)
            if len(vif_indices) > 0:
                print(lineno(), X_real.shape)
                X_real = X_real[:, vif_indices]
                cols_inputs = [cols_inputs[i] for i in vif_indices]
            print('max vif', max(vif))

        variance_threshold = False
        if variance_threshold is True: #http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
            from sklearn.feature_selection import VarianceThreshold
            sel = VarianceThreshold() # gets rid of unchanging variables.
            sel.fit(X_real)  # fit the inputs
            selected_feature_indices = sel.get_support(True)  # get indices of those selected
            print(lineno(), 'selected features', len(selected_feature_indices), selected_feature_indices)
            X_real = sel.transform(X_real)  # transform the matrix
            cols_inputs = [cols_inputs[i] for i in selected_feature_indices.tolist()]  # select the names based on selected features
            print(lineno(), cols_inputs)
            print(X_real.shape)

        f_scores = False
        if f_scores is True: #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
            feature_scores = []
            vars_in = X_real.shape[1]
            for i in range(Y_real.shape[1]):
                f_test, _ = f_regression(X_real, Y_real[:, i])
                feature_scores.append(f_test)
            df_feature_scores = pd.DataFrame(list(map(list, zip(*feature_scores))), columns=cols_outputs, index=cols_inputs)  # transpose list of lists and make a dataframe of it
            df_feature_scores['max_f'] = df_feature_scores.max(axis=1)
            df_scores = df_feature_scores[df_feature_scores['max_f'] > 1]
            print('no. variables removed by f_scores', vars_in-df_scores.shape[0])
            cols_inputs = df_scores.index.tolist()

        mutual_info_scores = False
        if mutual_info_scores == True:
            mi_scores = []
            vars_in = X_real.shape[1]
            for i in range(Y_real.shape[1]):
                mi = mutual_info_regression(X_real, Y_real[:, i])
                mi_scores.append(mi)
            df_mi_scores = pd.DataFrame(list(map(list, zip(*mi_scores))), columns=cols_outputs, index=cols_inputs)  # transpose list of lists and make a dataframe of it
            df_mi_scores['total_mi'] = df_mi_scores.max(axis=1)
            print(df_mi_scores['total_mi'])
            df_scores = df_mi_scores[df_mi_scores['total_mi'] > .01]
            print('no. variables removed by mi_scores', vars_in - df_scores.shape[0])
            cols_inputs = df_scores.index.tolist()

        df = df_in[cols_inputs]
        print(df.shape)
        X_real = df.as_matrix()


    #exit()
    # scale the data?
    #todo http://scikit-learn.org/stable/modules/preprocessing.html

    return X_real, cols_inputs

def ConvergenceCorrelations(df_input, df_output, building_label):
    ## PLOT Check convergence of Correlations coefficients over model iterations ##


    df_stdcorr = pd.DataFrame()
    output_variable = 4  # which output variable (energy use) to look at
    for i in range(101, 110, 1):  # looking at different input variables
        df_std = pd.DataFrame()
        spearman_list = []
        iterations = []
        for j in range(9, 300, 10):
            iterations.append(j)
            spearman_list.append(spearmanr(X_real[:j, i], Y_real[:j, output_variable])[0])  # last 0 signifies the rho value (1 = p-value)
        s1 = pd.Series(spearman_list)  # put Correlations in series
        df_std = pd.concat([df_std, s1], axis=0)  # combine empty dataframe with Correlations from series
        df_std.columns = [cols_outputs[output_variable] + '_' + cols_inputs[i]]  # name column
        df_stdcorr = pd.concat([df_stdcorr, df_std], axis=1)  # combine multiple Correlations

    s_it = pd.Series(iterations, name='iterations')
    df_stdcorr = pd.concat([df_stdcorr, s_it], axis=1)
    df_stdcorr.set_index('iterations', inplace=True)

    ax1 = df_stdcorr.plot(title='Spearman Correlation', fontsize=9)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * .6, box.height])

    ax1.legend(bbox_to_anchor=(1, .5), loc='center left')
    ax1.set_ylabel('Correlation')
    ax1.set_xlabel('Iterations')


# Run the surrogate learning process with different sample sizes to see the learning rate.
def SurrogateModel(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, ):
    print(lineno(), 'Xtrain, Ytrain', X_train.shape, Y_train.shape)

    np.random.seed(0)

    if for_sensitivity is True:
        model_sensitivity = '_sensitivity'
        if end_uses is True:
            model_sensitivity = model_sensitivity + '_enduses'
            end_use = '_end_uses'
        else:
            end_use = ''
    else:
        model_sensitivity = ''
        if end_uses is True:
            model_sensitivity = model_sensitivity + '_enduses'
            end_use = '_end_uses'
        else:
            end_use = ''

    lr = LinearRegression()
    lasso = Lasso()
    rr = Ridge()
    pls = PLSRegression(n_components=X_train.shape[1], scale=False, max_iter=1000)
    knn1 = KNeighborsRegressor(X_train.shape[1] if X_train.shape[1] > len(cols_inputs) else 5, weights='distance')
    nn = MLPRegressor(hidden_layer_sizes=(X_train.shape[1],), solver='lbfgs', activation='identity',  max_iter=1000, tol=.0001)

    rf = RandomForestRegressor()
    ts = TheilSenRegressor()
    ransac = RANSACRegressor()
    hr = HuberRegressor()


    # While many algorithms (such as SVM, K-nearest neighbors, and logistic regression) require features to be normalized, intuitively we can think of Principle Component Analysis (PCA) as being a prime example of when normalization is important.

    # do not support multivariate regression, use MultiOutputRegressor
    bayesm = BayesianRidge()

    svrm = GridSearchCV(SVR(kernel='linear'), param_grid={"C": [1, 10, 100],})
    #svrm = SVR(kernel='rbf', C=10, gamma=0.0001, epsilon=0.01)
    svrm = LinearSVR(C=30, epsilon=1, max_iter=10000)

    # krr = GridSearchCV(KernelRidge(), cv=5, param_grid={"alpha": [1e0, 1e-1, 1e-2, 1e-3],
    #                                                     "kernel": [ExpSineSquared(l, p)
    #                                                                for l in np.logspace(-2, 2, 10)
    #                                                                for p in np.logspace(0, 2, 10)]})

    krr = GridSearchCV(KernelRidge(coef0=1, degree=3, gamma=None, kernel='linear', kernel_params=None), param_grid={'alpha':[0.1, 1, 5], })

    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
    gpr = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=9, normalize_y=False)

    #("rr", rr),  ("k-NN", knn), ("NN", nn), ("RF", rf), ("BR", bayesm), ("Lasso", lasso), ("GPR", gpr), ("LR", lr)
    #("ts", ts), ("ransac", ransac), ("hr", hr) perform as well as Linear Regression / OLS
    #models = [("rr", rr), ("Lasso", lasso), ("NN", nn), ("pls", pls)]

    models = [("pls", pls), ("rr", rr), ("Lasso", lasso), ("krr", krr)]#, ("SVR", svrm),]#  ("NN", nn)
    #models = [("pls", pls)]

    #todo even though LR is quite accurate, it will give very high coefficient as it is overfitting the model, specifically for the overtime_multiplier inputs, RR seems to give more stable values.
    x=0

    y_pos = np.arange(len(models)) # no. of models/methods i am using.
    r2_list, mse_list, time_list, mae_list, evar_list = [], [], [], [], [] # fill lists with evaluation scores

    model_names = []

    print(lineno(), 'columns', cols_outputs)
    for name, model in models:
        print(name)
        model_names.append(name)
        stime = time.time()
        if model in {bayesm, svrm, gpr, rf}: # use multioutputregressor for these methods
            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            #                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            # surrogate = GridSearchCV(svrm, tuned_parameters)
            if model in {svrm}:
                #scaler = RobustScaler(quantile_range=(25, 75))
                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.fit_transform(X_test)

                model = MultiOutputRegressor(model)
                model.fit(X_train_scaled, Y_train)
                prediction = model.predict(X_test_scaled)
            else:
                model = MultiOutputRegressor(model)
                model.fit(X_train, Y_train)
                prediction = model.predict(X_test)

            if write_model == True:
                joblib.dump(model, DataPath_model_real+name + '_' + time_step + model_sensitivity +  '_model.pkl')
            else:
                print(lineno(), 'write_model set to:', write_model)

        elif model in {lr, rr, nn, lasso, nn, pls, krr, knn1}:
            if model in {nn, knn1}:
                #scaler = StandardScaler(copy=True, with_mean=True, with_std=True) #copy=True, with_mean=True, with_std=True
                scaler = RobustScaler(quantile_range=(25, 75))

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.fit_transform(X_test)
                model.fit(X_train_scaled, Y_train)
                prediction = model.predict(X_test_scaled)
            else:
                model.fit(X_train, Y_train)
                prediction = model.predict(X_test)

            if write_model == True:
                joblib.dump(model, DataPath_model_real+name + '_' + time_step + model_sensitivity + '_model.pkl')
            else:
                print(lineno(), 'write_model set to:', write_model)

        # timing / scoring
        print(lineno(), prediction[0])
        print('R^2 = ', r2_score(Y_test, prediction, multioutput='uniform_average'))

        raw_mse = mean_squared_error(Y_test, prediction, multioutput='raw_values')
        raw_mae = mean_absolute_error(Y_test, prediction, multioutput='raw_values')  # multioutput='raw_values' will give the error for all 31days*8enduses (248, )
        #TODO I should differentiate MSE between monthly and hourly response variables.

        time_list.append("{0:.4f}".format(((time.time() - stime))))
        r2_list.append("{0:.4f}".format((r2_score(Y_test, prediction, multioutput='uniform_average'))))
        mse_list.append("{0:.4f}".format((mean_squared_error(Y_test, prediction))))
        evar_list.append("{0:.4f}".format((explained_variance_score(Y_test, prediction))))
        mae_list.append("{0:.4f}".format((mean_absolute_error(Y_test, prediction))))

        abs_diff = [(j - i)**2 for i, j in zip(Y_test, prediction)]
        mse_abs_diff = [np.mean(i)/len(Y_test) for i in abs_diff] # the mean squared error for each iteration separately



    print(lineno(), 'learning times [seconds]', time_list)
    print(lineno(), 'r2', r2_list)

    df_scores = pd.DataFrame(np.column_stack([[x[0] for x in models], mae_list]),columns=['Algorithms','Explained Variance']) #, 'Mean squared error $\mathregular{[kWh/m^{2}a]}$'
    df_r2 = pd.DataFrame(np.column_stack([[x[0] for x in models], r2_list]), columns=['Algorithms', '$\mathregular{r^{2}}$'])
    df_scores.set_index('Algorithms', inplace=True)
    df_r2.set_index('Algorithms', inplace=True)
    df_scores= df_scores.astype(float) #change data back to floats..
    df_r2 = df_r2.astype(float)

    return mse_list, df_r2, df_scores, Y_test, prediction

def CreateElements(mus, sigmas, lower_limits, upper_limits, dist):  # create input variables
    def trunc_gauss(mu, sigma, bottom, top):
        a = random.gauss(mu, sigma)
        while (bottom <= a <= top) == False:
            a = random.gauss(mu, sigma)
        return a

    elements = []
    for x in range(len(mus)):
        if dist[x] == 'norm':
            elements.append(trunc_gauss(mus[x], sigmas[x], lower_limits[x], upper_limits[x]))

            #elements.append(random.gauss(mus[x], sigmas[x]))
        elif dist[x] == 'unif':
            elements.append(random.uniform(lower_limits[x], upper_limits[x]))
        elif dist[x] == 'disc':
            elements.append(random.randrange(int(lower_limits[x]), int(upper_limits[x]+1), int(1))) #integer discrete value

    #print([i - j for i,j in zip(elements, lower_limits)])

    if len(mus) != len(elements):
        raise ValueError('Number of mus, longer than input parameters in CreateElements')


    return elements

def mutGaussianWithLimits(individual, mus, sigmas, lower_limits, upper_limits, dist, indpb):
    """ Same as CreateElements, but now called upon during optimisation for mutation"""
    def trunc_gauss(mu, sigma, bottom, top):
        a = random.gauss(mu, sigma)
        while (bottom <= a <= top) == False:
            a = random.gauss(mu, sigma)
        return a

    for i in range(len(mus)):
        if random.random() < indpb:
            if dist[i] == 'norm':
                individual[i] = trunc_gauss(mus[i], sigmas[i], lower_limits[i], upper_limits[i])
            elif dist[i] == 'unif':
                individual[i] = random.uniform(lower_limits[i], upper_limits[i])
            elif dist[i] == 'disc':
                individual[i] = random.randrange(lower_limits[i], upper_limits[i] + 1, mu[i])  # integer discrete value
                print(individual[i])
                exit()
                #stepsize in mu
    return individual,

def UniformWithLimits(individual, lower_limits, upper_limits, indpb):
    size = len(individual)
    for i, l, u in zip(range(size), lower_limits, upper_limits):
        if random.random() < indpb:
            individual[i] = random.uniform(l, u)
    #print(len(individual), individual)
    return individual,

def calculate(individual, for_sensitivity):  # predict using input variables and pkl model
    model = ['LR']
    #("PLS", pls), ("SVR", svrm), # take too long
    #("rr", rr),  ("k-NN", knn), ("NN", nn), ("RF", rf), ("BR", bayesm), ("Lasso", lasso), ("GPR", gpr), ("LR", lr)
    #("ts", ts), ("ransac", ransac), ("hr", hr) perform as well as Linear Regression / OLS
    #models = [("rr", rr), ("BR", bayesm), ("Lasso", lasso), ("LR", lr), ("ts", ts), ("ransac", ransac), ("hr", hr)]
    if for_sensitivity is True:
        if end_uses is True:
            model_path =  '_sensitivity' + '_enduses_model'
        else:
            model_path =  '_sensitivity' + '_model'
    else:
        if end_uses is True:
            model_path =  '_enduses_model'
        else:
            model_path =  '_model'

    #print(lineno(), DataPath_model_real + model[0]+'_' + time_step + model_path + '.pkl')
    model = joblib.load(DataPath_model_real + model[0]+'_' + time_step + model_path + '.pkl')

    individual = np.array(individual).reshape((1, -1))
    prediction = model.predict(individual)[0]

    return prediction

def EvaluateObjective(individual):
    prediction = calculate(individual, for_sensitivity)
    prediction = [i for i in prediction]
    weights = [(i) / sum(targets) for i in targets]
    #todo weights probably doesn't work with hourly values typical weekdays
    rmse = sum(((i - j) ** 2) / len(targets) for i, j in zip(targets, prediction))  # minimise the total

    #rmse = sum(w*((i - j) ** 2)/len(targets) for w, i, j in zip(weights, targets, prediction)) # minimise the total


    #rmse = np.sqrt((sum((i - j)**2 for i, j in zip(targets, prediction)) / len(targets))) #minimise the total
    #rmse = tuple(((i - j) ** 2 for i, j in zip(targets, prediction))) #minimise each objective
    # cvrmse = np.sqrt((sum((i - j) ** 2 for i, j in zip(targets, prediction)) / len(targets))) / np.average(targets) * 100

    return rmse,

# Load the pickle surrogate model made by 'PKL_SurrogateModel.py' and calculate with new inputs
# def PKL_SurrogateModelSobolSamples(samples, mus, sigmas, lower_limits, upper_limits):
#     individuals = []
#     predictions = []
#     print('samples shape', samples.shape)
#     for x in range(samples.shape[0]):
#         individual = samples[x]
#         #todo limit to lower and upper limit
#         for i in range(len(individual)):
#             while (lower_limits[i] <= individual[i] <= upper_limits[i]) == False:
#                 individual[i] = random.gauss(mus[i], sigmas[i])
#
#         individuals.append(individual)
#         prediction = calculate(individual, for_sensitivity)
#         predictions.append(prediction)
#
#     X_real = np.matrix(individuals)
#     Y_real = np.matrix(predictions)
#
#     print('X_real and Y_real shapes', X_real.shape, Y_real.shape)
#
#     return (X_real, Y_real)

def PKL_SurrogateModel(samples, mus, sigmas, lower_limits, upper_limits, dist):
    individuals = []
    predictions = []

    for x in range(len(samples)):
        individual = CreateElements(mus, sigmas, lower_limits, upper_limits, dist)
        individuals.append(individual)
        prediction = calculate(individual, for_sensitivity)
        predictions.append(prediction)

    X_real = np.matrix(individuals)
    Y_real = np.matrix(predictions)

    print('X_real and Y_real shapes', X_real.shape, Y_real.shape)

    return (X_real, Y_real)

# Sobol does not work for the test data as it is sampled by LHS and it needs to be sampled by saltelli sampling (which creates many more variables), however it can be done with the surrogate model instead.
def SalibSensitivityAnalysis(time_step, cols_inputs, cols_outputs, inputs_basecase, mus, sigmas, dist, lower_limits, upper_limits, targets, include_sobol, method, building_abr):
    # https://media.readthedocs.org/pdf/salib/latest/salib.pdf | http://salib.github.io/SALib/ # http://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    # http://keyboardscientist.weebly.com/blog/sensitivity-analysis-with-salib

    D = len(cols_inputs)  # no. of parameters
    N = 50 # size of an initial monte carlo sample, 2000 simulations?
    real_cols = cols_inputs  # columns names of the parameters

    print(D * N, 'for Fast, Delta and DGSM')  # (N*D) = [Fast, Delta, DGSM]
    print((D + 1) * N, 'for Morris')  # (D + 1)*N = [Morris, ]
    print((D + 2) * N, 'for Sobol first order')
    print(N * (2 * D + 2), 'for Sobol second order')  # , (D+2)*N for first order = [Sobol, ]
    print(2 ** (round(np.sqrt(150), 0)), 'for fractional fast')  # (2**D) < N = [FF, ] (so if 300paramaters N = 2^9 > 300)

    if method == 'sobol':
        salib_sigmas = [0.00000011 if i == 0 else i for i in sigmas]  # because salib doesn't allow 0 values for std. dev
        salib_mus = [0.0000001 if i == 0 else i for i in mus]  # because salib doesn't allow 0 values for std. dev
        norm_bounds = pd.concat([pd.DataFrame(salib_mus), pd.DataFrame(salib_sigmas)], axis=1)
        norm_bounds = norm_bounds.values.tolist()  # turn into list [[mu, sigma], [mu, sigma] ...]
        print(norm_bounds)
        print(len(['norm' for i in range(D)]), ['norm' if i > 0 else 'unif' for i in sigmas])

        problem = {'num_vars': D, 'names': real_cols, 'bounds': norm_bounds, 'dists': ['norm' if i == 'unif' else i for i in dist]}  # unif for those that are 0

        second_order = True
        if second_order == True:
            samples = N * (2 * D + 2)
            sal_inputs = saltelli.sample(problem, N, calc_second_order=True)
        else:
            samples = N * (D + 2)
            sal_inputs = saltelli.sample(problem, N, calc_second_order=False)

        print(sal_inputs)
        print('no of samples', samples)

    if method == 'morris':
        print('morris')
        norm_bounds = pd.concat([pd.DataFrame(lower_limits), pd.DataFrame(upper_limits),], axis=1)
        #print(lineno(), norm_bounds.values.flatten().tolist())
        #print([[-np.pi, np.pi] * D])

        problem = {'num_vars': D, 'names': real_cols, 'bounds': [norm_bounds.values.flatten().tolist()]}
        samples = (D+1)*N

        sal_inputs = SALib.sample.morris.sample(problem, D+1, num_levels=4, grid_jump=2)

    print(len(sal_inputs))
    X_real, Y_real = PKL_SurrogateModel(sal_inputs, mus, sigmas, lower_limits, upper_limits, dist)

    df_inputs = pd.DataFrame(X_real, columns=[cols_inputs])
    df_outputs = pd.DataFrame(Y_real, columns=[cols_outputs])

    df_outputs = df_outputs / (3600*1000) / floor_area

    ## Analyze Sobol
    df_sobol_first = pd.DataFrame()
    df_sobol_total = pd.DataFrame()
    df_morris_mu = pd.DataFrame()
    df_morris_sigma = pd.DataFrame()
    if include_sobol is True:
        #If calc_second_order is False, the resulting matrix has N * (D + 2) rows, where D is the number of parameters. If calc_second_order is True, the resulting matrix has N * (2D + 2) rows.

        if method == 'sobol':
            print('sobol')
            for i, v in enumerate(df_outputs.iloc[:,:].columns):
                Si_sobol = sobol.analyze(problem, df_outputs[v].values, calc_second_order=second_order)
                df_first = pd.DataFrame(Si_sobol['S1'].tolist(), index=cols_inputs, columns=[str(v)])
                df_total = pd.DataFrame(Si_sobol['ST'].tolist(), index=cols_inputs, columns=[str(v)])
                df_sobol_first = pd.concat([df_sobol_first, df_first], axis=1)
                df_sobol_total = pd.concat([df_sobol_total, df_total], axis=1)
            print(df_sobol_first.head())

        elif method == 'morris':
            print('morris')
            #print( df_inputs.iloc[0,:].values, df_outputs.iloc[0,:].values)
            #print(df_inputs.head())
            print(df_inputs.values.shape, df_outputs.values.shape)

            for i,v in enumerate(df_outputs.iloc[:,:].columns):
                Si_morris = SALib.analyze.morris.analyze(problem, df_inputs.iloc[:,:].values, df_outputs.iloc[:,i].values)
                #df__mu = pd.DataFrame(Si_morris['mu'].tolist(), index=cols_inputs, columns=[str(v)])
                df__mu_star = pd.DataFrame(Si_morris['mu_star'].tolist(), index=cols_inputs, columns=[str(v)])
                df__sigma = pd.DataFrame(Si_morris['sigma'].tolist(), index=cols_inputs, columns=[str(v)])
                df_morris_mu = pd.concat([df_morris_mu, df__mu_star], axis=1)
                df_morris_sigma = pd.concat([df_morris_sigma, df__sigma], axis=1)
            #Output samplmes must be multiple of D+1 (parameters)
            #print(Si_morris['mu'])


        # Si_delta = delta.analyze(real_problem, X_real, Y_real[:,0])
        # print(Si_delta['S1'])
        # print(Si_delta['delta'])

        # Si_fast = fast.analyze(real_problem, Y_real[:,0])
        #  Output sample must be multiple of D (parameters)
        # print(Si_fast['S1'])

        # Si = dgsm.analyze(real_problem, X_real, Y_real[:,0])
        # print(Si['delta']) # Returns a dictionary with keys ‘delta’, ‘delta_conf’, ‘S1’, and ‘S1_conf’

        # Si_ff = ff.analyze(real_problem, X_real, Y_real[:,0])
        # The resulting matrix has D columns, where D is smallest power of 2 that is greater than the number of parameters.
        # print(Si_ff['ME'])

        # DUMMY SOBOL ##
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
        #test_sobol()

    return X_real, Y_real, df_inputs, df_outputs, df_sobol_first, df_sobol_total, df_morris_mu, df_morris_sigma



def OptimisationAlgorithm(toolbox, X_real):
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

    # print(lineno(), X_real.shape)
    # print(lineno(), pop[0])
    #
    # pd.DataFrame(pop).to_csv(DataPath_model_real+'pop.csv')
    # print(lineno(), 'test an individual from optimisation data', [int(i) for i in calculate(pop[0], for_sensitivity)])

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

    df_calibrated = pd.DataFrame()
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        # print(pop), print(len(pop[0]))
        # print([ind.fitness.valid for ind in pop if ind.fitness.valid])

        offspring = tools.selTournamentDCD(pop, len(pop))  # only works with "select" to NSGA-II
        #offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]  # Clone the selected individuals

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):  # Crossover and mutate offspring
            # print('inds', ind1, ind2)
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)  # crossover randomly chosen individual within the population

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
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, len(offspring))

        # fits = [ind.fitness.values[0] for ind in pop]
        best_ind = tools.selBest(pop, 1)[0]
        best_inds.append(best_ind)  # add the best individual for each generation
        best_inds_fitness.append(best_ind.fitness.values)

        record = stats.compile(pop)
        logbook.record(gen=gen, inputs=[int(e) for e in best_ind.fitness.values], **record)
        hof.update(pop)

        potential_individuals = tools.selBest(pop, int(POPULATION_SIZE/2)) # select k best individuals in populations
        if gen % 20 == 0:

            print(gen, 'fitness', [int(e) for e in best_ind.fitness.values],  int(abs(sum(targets) - sum(calculate(best_inds[-1], for_sensitivity)))), 'best_inds',
                  [int(e) for e in calculate(best_inds[-1], for_sensitivity).tolist()], 'targets', [int(i) for i in targets], individual)
            #print(len(calculate(best_inds[-1], for_sensitivity)))
            # with open("logbook.pkl", "wb") as lb_file: pickle.dump(logbook, lb_file)

        #collect individuals that fit mbe and cvrmse criteria.
        if gen > 2:
            for individual in potential_individuals:
                prediction = calculate(individual, for_sensitivity)
                prediction = [i for i in prediction]
                #cvrmse_month = np.sqrt(sum(((i - j) ** 2) / len(df_month) if i > 0 else 0 if j > 0 else 0 for i, j in zip(df_month, df_pred_month))) / np.average(df_month) * 100
                cvrmse = np.sqrt(sum(((i - j) ** 2) / len(targets) for i, j in zip(targets, prediction))) / np.average(targets) * 100
                mbe = sum([i - j for i, j in zip(prediction, targets)]) / sum(prediction) * 100

                if cvrmse < 2 and mbe < 3 and mbe > -3:
                    df_ind = pd.Series(individual)
                    if df_calibrated[(df_calibrated == df_ind)].all(axis=1).any() != True: #check if individual already exists in dataframe
                        df_calibrated = df_calibrated.append(df_ind, ignore_index=True)
                        #print(cvrmse, mbe, prediction)
    print(cvrmse, mbe, prediction)
    print(lineno(), 'number of calibrated models', len(df_calibrated))

    # best_ind = tools.selBest(pop, 1)[0] # best_ind = max(pop, key=lambda ind: ind.fitness)
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # print(logbook.stream)

    return pop, hof, logbook, best_inds, best_inds_fitness, df_calibrated

def OptimiseModel(time_step, targets, mus, sigmas, lower_limits, upper_limits, dist, cols_outputs, X_real):

    print(lineno(), 'no. of objectives', len(targets))
    print('weights normalized to target', tuple([-(i) / sum(targets) for i in targets]))
    print('equal weights', tuple([-1.0 for i in targets]))
    
    creator.create('Fitness', base.Fitness, weights= tuple([-1.0 for i in targets]))
    creator.create('Individual', array.array, typecode='d', fitness=creator.Fitness)  # set or list??

    toolbox = base.Toolbox()
    toolbox.register('expr', CreateElements, mus, sigmas, lower_limits, upper_limits, dist) ## using custom function for input generation
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # TODO try different selection criteria/mutate/mate
    toolbox.register('mate', tools.cxTwoPoint)
    #toolbox.register("mate", tools.cxUniform, indpb=INDPB)
    #toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_limits, up=upper_limits, eta=20.0)

    #toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_limits, up=upper_limits, eta=20.0, indpb=INDPB) #TODO can't divide by zero, need to remove variables that are 0
    #toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigmas, indpb=INDPB)  # sigmas are set to a sequence, #TODO retrieve the right values, they are now based on 1/10 of some initial sample
    toolbox.register('mutate', mutGaussianWithLimits, mus=mus, sigmas=sigmas, lower_limits=lower_limits, upper_limits=upper_limits, dist=dist, indpb=INDPB)
    #toolbox.register('mutate', UniformWithLimits, lower_limits, upper_limits, indpb=INDPB)

    toolbox.register('select', tools.selNSGA2)
    #toolbox.register('select', tools.selSPEA2)

    toolbox.register('evaluate', EvaluateObjective)  # add the evaluation function
    # toolbox.register("evaluate", benchmarks.zdt1)

    pop, hof, logbook, best_inds, best_inds_fitness, df_calibrated = OptimisationAlgorithm(toolbox, X_real)

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

    print(best_inds[-1])
    print(len(best_inds))
    print('cols_targets', cols_outputs)
    print('best individual prediction vs. targets')
    print('best    ', [int(e) for e in calculate(best_inds[-1], for_sensitivity).tolist()])
    print('targets ', targets)
    print('diff    ', [i - j for i, j in zip([int(e) for e in calculate(best_inds[-1], for_sensitivity).tolist()], targets)])
    print('best    ', [i/floor_area for i in [int(e) for e in calculate(best_inds[-1], for_sensitivity).tolist()]])
    print('targets ', [i/floor_area for i in targets])

    # print(logbook)
    # print(pop)

    return best_inds, best_inds_fitness, df_calibrated

def main(end_uses):
    np.random.seed(0)
    if generate_inputsoutputs: #create outputs file (combine each simulation results file into one)
        df_pred, runs, df_runs, df_enduses, df_weather, df_runs_weekdays = ReadRuns(building, base_case, calibrated_case, compare_models, c_file_name, floor_area, building_harddisk, building_abr, time_step, write_data, NO_ITERATIONS, end_uses, include_weekdays, for_sensitivity)

        if time_step == 'month' and end_uses is False:
            if include_weekdays is True:
                runs = pd.merge(runs, df_runs_weekdays, left_index=True, right_index=True)

        if time_step == 'month' and end_uses is True:
            end_uses_list = list(runs.columns.levels[0])
            if include_weekdays is True:#
                time_list = [i for i in df_runs_weekdays.columns.levels[1]]
                runs = pd.merge(runs, df_runs_weekdays, left_index=True, right_index=True)

        print(runs.head())
        print(runs.columns)
        print(runs.columns.tolist())

        # saving outputs to hdf file because then I don't have to read the eplusmtr files each time.
        runs.to_hdf(DataPath_model_real + 'inputs_outputs_'+time_step+'_'+str(end_uses)+'_'+str(include_weekdays)+'.hdf', 'runs', mode='w')
        exit()

    else:
        runs = pd.read_hdf(DataPath_model_real + 'inputs_outputs_'+time_step+'_'+str(end_uses)+'_'+str(include_weekdays)+'.hdf', 'runs')
        #print(runs.head(3))
        #print(runs.shape)
        #print(runs.columns.levels[0])
        level_0 = [i for i in runs.columns.levels[0]]
        level_1 = [i for i in runs.columns.levels[1]]
        cols_outputs = [str(i)+ ' ' +str(j) for i in level_0 for j in level_1] #combine levels)

        print(len(cols_outputs), cols_outputs)

    #select inputs and outputs and run feature selection if True
    Y_real = runs.as_matrix()
    X_real, cols_inputs = ImportOutputs(DataPath_model_real, inputs, Y_real, cols_outputs, run_numbers, feature_selection)  # real input data

    if build_surrogate_model is True:
        # i can change the selection of runs for a specific end-use to train on that, this will help me identify where the meta-model is having trouble learning behaviour.

        #cols_outputs = cols_outputs[Y_real.shape[1]*2:3*Y_real.shape[1]]
        X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real)  # randomly split data to create test set

        ### BUILDING THE SURROGATE MODEL USING TRAINING DATA
        if plot_progression is True:
            PlotSurrogateLearningRate(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, ) # this creates the model mutiple times.

        mse_list, df_r2, df_scores, Y_test, prediction = SurrogateModel(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, )

        #PlotValidateModelFit(Y_test, prediction, cols_outputs, time_step)
        PlotSurrogateModelPerformance(df_scores, df_r2, building_label)


        if write_data is True:
            X_real_data = np.vstack((cols_inputs, X_real))
            Y_real_data = np.vstack((cols_outputs, Y_real))
            input_outputs = np.hstack((X_real_data, Y_real_data))
            with open(DataPath_model_real + 'input_outputs_' + time_step + '.csv', "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(input_outputs)

    print('runs shape', runs.shape)
    if sensitivity_analysis_predictions is True:
        ### SENSITIVITY ANALYSIS OF REAL DATA
        Y_real = runs.as_matrix()
        print(lineno(), X_real.shape, Y_real.shape, )

        df_corr, df_corr_standardized, df_corr_spearman, df_corr_pearson = Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step, metamodel_sensitivity)

        #print(lineno(), df_corr.head())
        #PlotCorrelations(df_corr, cols_outputs)
        df_outputs = pd.DataFrame(Y_real, columns=[cols_outputs])
        HeatmapCorrelations(df_corr, df_outputs, cols_outputs, include_sobol, building_label)

    if metamodel_sensitivity is True or function_optimisation is True:

        df_inputs = pd.read_csv(DataPath_model_real + inputs_basecase, index_col=0, header=0)
        df_inputs = df_inputs[cols_inputs] # select only those after feature selection filter!

        mus = df_inputs.loc['mu',:].values.tolist()
        mus = [float(i) for i in mus]
        if evidenced_range is True:
            perc_var = df_inputs.loc['perc_var_evidence',:].values.tolist()
        else:
            perc_var = df_inputs.loc['perc_var',:].values.tolist()

        perc_var = [float(i) for i in perc_var]
        sigmas = [i*j if i*j>0 else 0 for i,j in zip(mus, perc_var)]
        lower_limits, upper_limits = [i-(j*3) for i,j in zip(mus, sigmas)], [i+(j*3) for i,j in zip(mus, sigmas)]
        sigmas_optimise = [float(i) for i in sigmas] # to see what happens when variables are allowed out of bounds.
        dist = df_inputs.loc['distribution',:].tolist()

        print(lineno(), 'mus', mus)
        print(lineno(), 'sigmas', sigmas)
        print(lineno(), 'lower', lower_limits)
        print(lineno(), 'upper', upper_limits)
        print(lineno(), 'dist', dist)


    if metamodel_sensitivity is True:

        X_real, Y_real, df_inputs, df_outputs, df_sobol_first, df_sobol_total, df_morris_mu, df_morris_sigma = SalibSensitivityAnalysis(time_step, cols_inputs, cols_outputs, inputs_basecase, mus, sigmas, dist, lower_limits, upper_limits, targets, include_sobol, method, building_abr)

        # samples = 100
        # X_real, Y_real = PKL_SurrogateModel(samples, mus, sigmas, lower_limits, upper_limits, dist)
        # df_computed_inputs = pd.DataFrame(X_real, columns=[cols_inputs])
        # df_outputs = pd.DataFrame(Y_real, columns=[cols_outputs])

        #PlotParallelCoordinates(df_inputs, df_computed_inputs, df_outputs, building_label)

        #df_corr_standardized, df_corr_spearman, df_corr_pearson = Correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step, metamodel_sensitivity)
        # df_corr_standardized.to_csv(DataPath_model_real + 'df_corr_standardized.csv')
        # df_corr_spearman.to_csv(DataPath_model_real + 'df_corr_spearman.csv')
        # df_corr_pearson.to_csv(DataPath_model_real + 'df_corr_pearson.csv')
        # df_sobol_first.to_csv(DataPath_model_real + 'df_sobol_first.csv')
        # df_sobol_total.to_csv(DataPath_model_real + 'df_sobol_total.csv')
        df_morris_mu.to_csv(DataPath_model_real + 'df_morris_mu.csv')
        df_morris_sigma.to_csv(DataPath_model_real + 'df_morris_sigma.csv')

        #PlotCompareCoefficients(building_label)
        PlotCompareElementaryEffects(building_label)
        # PlotCalibratedSolutions(df_inputs, df_calibrated, building_label)
        # ScatterCorrelation(df_inputs, df_outputs, building_label, var_i = 'HW Boiler 1', var_o = 'Gas', )
        # ScatterCorrelation(df_inputs, df_outputs, building_label, var_i ='HW Boiler 2', var_o='Gas', )
        # ScatterCorrelation(df_inputs[['Office_LightSched_WeekProfile_TotalHours']], df_outputs[['Lights']], building_label)

        # PlotUncertainty(df_outputs, df_inputs, cols_outputs, time_step, end_uses, building_abr, building_label)
        # PlotSurrogateStandardError(X_real, Y_real, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, )
        # PlotSurrogateSpearmanConvergence(X_real, Y_real, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, )

        # HeatmapCorrelations(df_corr, df_outputs, cols_outputs, include_sobol, building_label)
        # HeatmapCorrelations(df_sobol_first, df_outputs, cols_outputs, include_sobol, building_label)

        # PlotSurrogateLearningRate(X_real, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, end_uses, for_sensitivity, plot_progression, write_model, write_data, )
        # PlotSobolIndices(Si_sobol, cols_outputs, cols_inputs,)

    if function_optimisation is True:
        print('optimising')
        df_computed_inputs = pd.DataFrame(X_real, columns=[cols_inputs])

        df_calibrated = pd.DataFrame()
        best_inds, best_inds_fitness, df_calibrated = OptimiseModel(time_step, targets, mus, sigmas_optimise, lower_limits, upper_limits, dist, cols_outputs, X_real)

        best_individual = pd.DataFrame(best_inds[-1].tolist(), index=cols_inputs, columns=['best_ind'])
        best_individual.T.to_csv(DataPath_model_real + 'best_individual'  + time_step + str(end_uses) + weekday_str + '.csv')

        if df_calibrated.empty:
            print('df_calibrated contains no solutions that fit the criteria')

            # df_calibrated = pd.read_csv(DataPath_model_real + 'CalibratedInputs_FromOptimisation' + time_step + str(end_uses) + '.csv', index_col=0, header=0)
            # df_calibrated.columns = cols_inputs
            # PlotCalibratedSolutionsBoxplot(df_inputs, df_computed_inputs, df_calibrated, lower_limits, upper_limits, building_label)  # boxplot

        else:
            df_calibrated.columns = cols_inputs
            df_calibrated.to_csv(DataPath_model_real + 'CalibratedInputs_FromOptimisation' + time_step + str(end_uses) + weekday_str + '.csv')

            #PlotCalibratedSolutionsBoxplot(df_inputs, df_computed_inputs, df_calibrated, lower_limits, upper_limits, building_label) # boxplot
            #PlotCalibratedSolutions(df_inputs, df_calibrated, building_label) # distribution plot

        # PlotPredictionGen()
        # PlotBestFit(cols_outputs, best_inds_fitness)
        PlotObjectiveConvergence(best_inds, cols_outputs, targets, floor_area, time_step, end_uses, building_label)  # shows each end use and how they change absolutely over the generations
        if include_weekdays:
            PlotObjectiveTimeConvergence(best_inds, cols_outputs, targets, floor_area, time_step, end_uses, building_label)  # shows each end use and how they change absolutely over the generations

    plt.show()

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
    InputVariables = ['inputs_MaletPlace11_08_16_11.csv', 'inputs_CentralHouse_222_08_11_15_21.csv', 'Inputs.csv', 'inputs_BH71_27_09_13_46.csv'] # 3000 inputs_CentralHouse_222_29_10_13_40
    InputVariablesBaseCase = ['x', 'inputs_basecase_CentralHouse.csv', 'inputs_basecase_17.csv', 'inputs_basecase_BH71.csv']
    FloorAreas = [9579, 5876, 1924, 1691]

    building_num = 1 # 0 = MPEB, 1 = CH, 2 = 17, 3 = 71
    base_case = False # only show a single run from the basecase, or multiple runs (will change both import and plotting)
    calibrated_case = False
    NO_ITERATIONS = 500
    time_step = 'month'  # 'year', 'month'
    end_uses = True # for month specifically (when True it will divide end-uses by month instead of aggregating to total)
    write_model = False # writes out the .pkl model.
    write_data = False # Writes out several .csv files some used for function optimisation script atm.
    plot_progression = True # this will plot the learning progression of the model when set to True.

    sensitivity_analysis_predictions = False # sensitivity analysis only for the parallel simulation runs

    generate_inputsoutputs = False #inupt and output hdf5 file for easy training.
    build_surrogate_model = True # run and build surrogate model, and sensitivity of real data
    include_weekdays = False # to include weekdays in the targets/runs for both surrogate model creation and/or calibration
    feature_selection = True  # feature selection for when building a new model
    evidenced_range = False # to use evidenced variability in the input parameters or a standard value (for the sigma).
    for_sensitivity = False # if to build/read the surrogate model for sensitivity analysis (include all end-uses even those not measured)
    metamodel_sensitivity = False # use the built model for sobol
    include_sobol = False
    method = 'sobol'  # 'sobol', 'morris'
    function_optimisation = False # optimise towards mesured data  #if function_optimisation is True: NO_ITERATIONS = 1  # need col names...

    if include_weekdays is True:
        weekday_str = 'hourly'
    else:
        weekday_str = ''

    compare_models = False
    c_file_name = ''
    print('no of iterations', NO_ITERATIONS)
    # OPTIMISATION INPUTS
    NGEN = 300  # is the number of generation for which the evolution runs
    POPULATION_SIZE = 24 # no of individuals/samples  For selTournamentDCD, pop size has to be multiple of four
    CXPB = 0.8  # is the probability with which two individuals are crossed
    MUTPB = 0.5 # probability of mutating the offspring
    INDPB = 0.05 # Independent probability of each attribute to be mutated
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
    DataPathImages = "C:/Users/" + getpass.getuser() + "/Dropbox/01 - EngD/01 - Thesis/02_Images/"
    DataPathSTM = 'C:/Users/' + getpass.getuser() + '/Dropbox/01 - EngD/07 - UCL Study/MonitoringStudyUCL/'

    not_run = []
    run_numbers = []
    for i in range(NO_ITERATIONS):
        file_name = 'eplusmtr' + str(format(i, '04')) + '.csv'
        if os.path.isfile(os.path.join(DataPath_model_real+'/Eplusmtr/', file_name)):
            run_numbers.append(i)
        else:
            not_run.append(str(format(i, '04')))

    print(run_numbers)
    #ImportOutputs(DataPath_model_real, inputs, run_numbers, feature_selection)

    if function_optimisation is True or metamodel_sensitivity is True or sensitivity_analysis_predictions is True:
        # READ DATA
        if building_num in {0, 1}:  # Does it have short term monitoring?
            df_stm = ReadSTM(DataPathSTM, building, building_num, write_data, datafile, floor_area)  # read short term monitoring
        else:
            df_stm = pd.DataFrame()

        if building_num in {1}:  # does it have separate gas use?
            df_gas = ReadGas(DataPath, building, building_num, write_data, datafile, floor_area)  # read gas data for Central House
        else:
            df_gas = pd.DataFrame()
        df, df_LVL1, df_floorsLP, df_mech, df_stm = ReadSubMetering(DataPath, building, building_num, building_abr, write_data, datafile, df_stm, floor_area)

        ### FUNCTION OPTIMISATION
        if time_step == 'year':
            if building_abr in {'71'}:
                df_LVL1 = df_LVL1.loc['01-01-14 0:00':'31-12-14']
            df_LVL1 = df_LVL1.sum(axis=0)
            if building_abr in {'CH'}:
                df_LVL1 = pd.concat([df_LVL1, df_gas.sum(axis=0)])
            print(lineno(), df_LVL1.head())

            targets = df_LVL1

        elif time_step == 'month':
            if end_uses is True:
                if building_abr in {'71'}:
                    df_LVL1 = df_LVL1.loc['01-01-14 0:00':'31-12-14 23:30']
                    if include_weekdays is True:
                        df_weekdays, df_weekdays_std = CreateAvgProfiles(df_LVL1.resample('H').sum(), day_type='three')
                    df_LVL1 = df_LVL1.resample('M').sum()
                if building_abr in {'CH'}:
                    if include_weekdays is True:
                        df_weekdays, df_weekdays_std = CreateAvgProfiles(df_LVL1.resample('H').sum(), day_type='three')

                    df_LVL1 = pd.concat([df_LVL1.resample('M').sum(), df_gas], axis=1)

                targets=[]
                for col in df_LVL1.columns:
                    targets.extend(df_LVL1[col].values.flatten().tolist())

                if include_weekdays is True:
                    df_weekday = pd.concat([df_weekdays.loc['weekday'], df_weekdays.loc['weekend']], axis=0)
                    targets.extend(df_weekday.values.flatten().tolist())

                # todo be careful here, don't change end-uses and months, compare to what the meta-model is trained on (runs df)
            else:
                if include_weekdays is True:
                    df_weekdays, df_weekdays_std = CreateAvgProfiles(df_LVL1.resample('H').sum(), day_type='three')

                if building_abr in {'71'}:
                    df_LVL1 = df_LVL1.loc['01-01-14 0:00':'31-12-14 23:30']
                    targets = df_LVL1.sum(axis=1).resample('M').sum().values.flatten().tolist()
                elif building_abr in {'CH'}:
                    targets = pd.concat([df_LVL1.resample('M').sum(), df_gas], axis=1).values.flatten().tolist()

                if include_weekdays is True:
                    df_weekday = pd.concat([df_weekdays.loc['weekday'], df_weekdays.loc['weekend']], axis=0)
                    targets.extend(df_weekday.values.flatten().tolist())

        #targets = [i for i in targets]
        print(lineno(), len(targets), 'targets', targets)

    # plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams.update({'font.size': 9})

    main(end_uses)





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