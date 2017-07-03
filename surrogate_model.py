import numpy as np
from numpy import linspace
import getpass
import time
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import csv
#import seaborn as sns
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

import pickle

def GaussianDistMultiVariable(df): # works only with 15min data!!
    print(len(df.shape))
    plt.figure(figsize=(16/2.54, 8/2.54))
    if len(df.shape) == 1: # if df only exists of one variable/columns
        kde = gaussian_kde(df)
        dist_space = linspace(min(df), max(df), 100)
        plt.plot(dist_space, kde(dist_space), label="occupied")

    elif len(df.shape)>1: # for multiple columns
        cols = df.columns.tolist()
        for i,v in enumerate(cols):
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b','#d62728','#9467bd','#aec7e8','#ffbb78','#98df8a','#c49c94','#ff9896','#c5b0d5','red','green','blue','black', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b','#d62728','#9467bd','#aec7e8','#ffbb78','#98df8a','#c49c94','#ff9896','#c5b0d5','red','green','blue','black']
            #print(df[v])
            kde = gaussian_kde(df[v])
            dist_space = linspace(min(df[v]), max(df[v]), 100)
            #print(dist_space)
            plt.plot(dist_space, kde(dist_space), label=v, color=colors[i])

    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.tight_layout()
    plt.legend(loc='best')
    plt.ylabel('Distribution')
    plt.xlabel('Energy [kWh/m2a]')
    plt.show()
def PlotAllArea(df):

    print('shape of df: ', len(df.shape))

    if len(df.shape) == 1: # if df only exists of one variable/columns
        df.plot(kind='area', stacked=True, color='#1f77b4', alpha=.5)

    elif len(df.shape)>1: # for multiple columns
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
                  '#c49c94', '#ff9896', '#c5b0d5']
        cols = df.columns.tolist()
        df.plot(kind='area', stacked=True, color=colors, legend='best', alpha=.5)

    plt.legend(loc='best')
    plt.ylabel('Energy [J]')
    plt.show()
def MultiBarPlot(df):
    if len(df.shape) == 1: # if df only exists of one variable/columns
        df.plot(kind='bar', stacked=True,legend='best', color='#1f77b4', alpha=.5)
    elif len(df.shape)>1: # for multiple columns
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
                  '#c49c94', '#ff9896', '#c5b0d5', 'red', 'green', 'blue', 'black']
        cols = df.columns.tolist()
        df.plot(kind='bar', stacked=True,legend='best', color=colors, alpha=.5)

    plt.gcf().autofmt_xdate()
    plt.ylabel('Energy (kWh)')
    plt.show()
def MultiBarBoxPlot(df):
    if len(df.shape) == 1: # if df only exists of one variable/columns
        df.plot.box()
    elif len(df.shape)>1: # for multiple columns
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
                  '#c49c94', '#ff9896', '#c5b0d5', 'red', 'green', 'blue', 'black']
        cols = df.columns.tolist()
        df.plot.box()

    plt.legend(loc='best')
    plt.ylabel('Energy (kWh)')
    plt.show()
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

def import_outputs(DataPath_model_real, TIME_STEP, NO_ITERATIONS):
    runs = pd.DataFrame()
    run_no = []
    done = False

    not_run = []
    run_numbers = []

    for i in range(NO_ITERATIONS):
        file_name = 'eplusmtr' + str(format(i, '04')) + '.csv'
        if os.path.isfile(os.path.join(DataPath_model_real, file_name)):
            run_numbers.append(i)
        else:
            not_run.append(str(format(i, '04')))

    print(run_numbers)
    print(len(run_numbers))
    print(not_run)
    print(len(not_run))

    list_iterations = []
    for i in range(NO_ITERATIONS):
        list_iterations.append(i)

    print('importing data at TIME_STEP:', TIME_STEP)
    for i in list_iterations:
        #print(str(format(i, '04')))
        file_name = 'eplusmtr'+str(format(i, '04'))+'.csv'
        if os.path.isfile(os.path.join(DataPath_model_real, file_name)):
            # todo change date_range here for yearly sim
            rng = pd.date_range(start='1/1/2017 01:00:00', end='01/01/2018', freq='H')

            #TODO if using actual weather file in comparison to real data, i have to set exact months isntead in rng.
            df = pd.read_csv(os.path.join(DataPath_model_real, file_name), header=0, index_col=0)
            df = df[48:] #exclude 2 design days
            df = df.set_index(rng)
            df.index = df.index - pd.DateOffset(hours=1) # !! set all hours back one, so that 01:00 = 00:00 and 24:00 becomes 23:00.
            # So the energy use in the first hour is set at 00:00. This is in line with the submetered data
            # Also, pandas works from 0-23 hours, it would otherwise count 24 towards the next day.

            plot_run = 1 # plotting single runs.
            if plot_run == 0:
                if i == 20: #plot which run
                    df = df[:24*7] #length 7 days
                    df = df.div(3600*1000*FLOOR_AREA)
                    PlotAllArea(df)
                    done = True
                    break

            if TIME_STEP == 'year': # with a full year of simulation, 1year*8enduses
                df_year = df.resample('A').sum()
                runs = pd.concat([runs, df_year], axis=0)

                if i == list_iterations[-1]:
                    cols = runs.columns.tolist()
                    #Rename the columns! # how to do so for month/day?
                    for i, v in enumerate(cols):
                        if 'THERMAL ZONE' not in cols[i]:  # exclude those meters that are thermal zone specific
                            rest = cols[i].split(':', 1)[0]  # removes all characters after ':'
                            rest = rest.replace('Interior', '')
                        else:
                            rest = cols[i].split(':', 1)[1]
                            rest = ''.join(cols[i].partition('ZONE:')[-1:])
                            rest = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", rest)  # remove text within symbols
                            rest = re.sub("[\(\[].*?[\)\]]", "", rest)  # remove symbols
                            rest = rest.strip()  # remove leading and trailing spaces
                            rest = rest.lower()
                        # print(rest)
                        runs.rename(columns={cols[i]: rest}, inplace=True)
                    runs.reset_index(drop=True, inplace=True)  # throw out the index (years)

            elif TIME_STEP == 'month':
                df_month = df.resample('M').sum()
                print(df_month)
                cols = df_month.columns.tolist()
                months = []

                for col in cols:
                    months.extend(df_month[col].tolist())

                df_months = pd.DataFrame(pd.Series(months))
                runs = pd.concat([runs, df_months.T], axis=0)

                if i == list_iterations[-1]:
                    months_index = df_month.index.tolist()
                    months_from_index = []
                    for month in months_index:
                        for col in cols:
                            months_from_index.append(str(month.month)+'_'+str(col))
                    print(months_from_index)

                    runs.reset_index(drop=True, inplace=True)
                    runs.columns = months_from_index

                    cols = runs.columns.tolist()
                    #Rename the columns! # how to do so for month/day?
                    for i, v in enumerate(cols):
                        if 'THERMAL ZONE' not in cols[i]:  # exclude those meters that are thermal zone specific
                            rest = cols[i].split(':', 1)[0]  # removes all characters after ':'
                            rest = rest.replace('Interior', '')
                        else:
                            rest = cols[i].split(':', 1)[1]
                            rest = ''.join(cols[i].partition('ZONE:')[-1:])
                            rest = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", rest)  # remove text within symbols
                            rest = re.sub("[\(\[].*?[\)\]]", "", rest)  # remove symbols
                            rest = rest.strip()  # remove leading and trailing spaces
                            rest = rest.lower()
                        # print(rest)
                        runs.rename(columns={cols[i]: rest}, inplace=True)
                    runs.reset_index(drop=True, inplace=True)  # throw out the index (years)

            elif TIME_STEP == 'day': # with a full year of simulation, 265days*8enduses
                df_day = df.resample('D').sum()
                cols = df_day.columns.tolist()
                days = []

                # transpose days per objective, have the days on the x-axis
                for col in cols:
                    days.extend(df_day[col].tolist())

                df_days = pd.DataFrame(pd.Series(days))
                runs = pd.concat([runs, df_days.T], axis=0)

                if i == list_iterations[-1]: # at last iteration, assign column names
                    days_index = df_day.index.tolist()
                    days_from_index = []
                    # Add day and end use string together
                    for day in days_index:
                        # days_from_index.append()
                        for col in cols:
                            days_from_index.append(str(day.month) + '-' + str(day.day) + str(col))

                    print(days_from_index)
                    print('length columns', len(days_from_index))

                    runs.reset_index(drop=True, inplace=True)  # throw out the index (years)
                    runs.columns = days_from_index  # rename columns

            elif TIME_STEP == 'hour':
                df_hour = df.resample('H').sum()
                runs = pd.concat([runs, df_hour], axis=0)

            if done == True:
                sys.exit('plotting single run and breaking')
                break

    # Set run numbers as index
    run_no = pd.DataFrame(pd.Series(run_numbers)) # create pandas dataframe of run numbers
    run_no.columns = ['run_no']
    print(run_no)
    runs = pd.concat([run_no, runs], axis=1, ignore_index=False) # prepend the run numbers to existing dataframe

    ## Output the energy predictions to .csv ##
    runs.to_csv(DataPath_model_real + 'runs_outputs_'+TIME_STEP+'.csv')
    runs.drop('run_no', axis=1, inplace=True)

    ## Check convergence of outputs over iterations ##
    plot_std = 0
    if plot_std is 1:
        print('standard error', sem(runs.iloc[:300]))
        df_std, df_stderr = pd.DataFrame(), pd.DataFrame()
        for i in range(9, 299, 10):
            s1 = pd.Series(runs.iloc[:i].std(), name=str(i))
            s2 = pd.Series(sem(runs.iloc[:i]), name=str(i))
            df_std = pd.concat([df_std, s1], axis=1)
            df_stderr = pd.concat([df_stderr, s2], axis=1)

        print(runs.iloc[:300].mean() - (sem(runs.iloc[:300]) * 1.96))
        print(runs.iloc[:300].mean() + (sem(runs.iloc[:300]) * 1.96))
        df_std = df_std.transpose()
        df_stderr = df_stderr.transpose()
        df_stderr.columns = cols_outputs

        ax1 = df_stderr.plot(figsize=(16/2.54, 8/2.54), title='Standard error')
        ax1.set_ylabel('MWh')
        ax1.set_xlabel('Iterations')

        #df_std[[3]].plot() ## use df_std[[0]] to call a single column

    runs = runs.div(3600*1000) # convert Joules to kWh #TODO include *FLOOR_AREA for per floor area
    #print(runs.head())
    ## BoxPlot multiple iterations in a boxplot to see distribution ##
    #ax2 = runs.plot.box()
    #ax2.set_ylabel('MWh')


    #MultiBarPlot(runs)
    #MultiBarBoxPlot(runs)
    #GaussianDistMultiVariable(runs.ix[:,0:7]) # to look at a single end-use, do runs[[col-no]], for several use, runs.ix[:,5:7]

    #runs.drop('WaterSyste', axis=1, inplace=True)
    cols_outputs = runs.columns.tolist()
    #print(runs.iloc[[20]]) # get row at index 20
    Y_real = runs.as_matrix()
    print('Y_real.shape', Y_real.shape)
    plt.show()

    return Y_real, cols_outputs, run_numbers

def import_inputs(DataPath_model_real, run_numbers):
    df = pd.read_csv(DataPath_model_real + 'inputs_CentralHouse_22225_06_11_49.csv', header=0) #TODO using backup file atm
    cols_inputs = df.columns.tolist()
    #print(["%.2d" % x for x in (np.arange(0, df.shape[0]))]) #zero pad a list of numbers
    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)
    X_real = df.ix[run_numbers]
    # X_real = df.ix[:,:]
    X_real = X_real.drop(df.columns[[0]], axis=1) #drop run_no column
    print('X_real shape', X_real.shape)
    X_real = X_real.as_matrix()

    return X_real, cols_inputs

def visualise_outputs(Y_real, cols_outputs):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']
    fig = plt.figure(0)
    ax1 = plt.subplot(111)
    print(Y_real.shape[0])
    ind = np.arange(Y_real.shape[0])
    bar_width = .3

    # TODO make a second figure to plot the scores
    #ax2.y_ticks(y_pos+bar_width /2, prediction.columns)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*.8, box.height])

    # plot actual data
    for i in range(len(cols_outputs)):
        ax1.bar(ind+bar_width, Y_real[:,i], bar_width, color=colors[i], label=cols_outputs[i])

    ax1.legend(bbox_to_anchor=(1,.5), loc='center left')    #http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    #ax2.legend(bbox_to_anchor=(1,.5), loc='center left')
    #plt.tight_layout()
    plt.show()

def correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, TIME_STEP):
    #fig = plt.figure(0)
    #ax = plt.subplot(111)

    print('no. variables', len(cols_inputs))
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
    df_corr.to_csv(DataPath_model_real + 'correlations' + TIME_STEP + '.csv')
    print(df_corr.shape)

    ## PLOT Check convergence of correlations coefficients over model iterations ##
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
            s1 = pd.Series(spearman_list) # put correlations in series
            df_std = pd.concat([df_std, s1], axis=0) #combine empty dataframe with correlations from series
            df_std.columns = [cols_outputs[output_variable]+'_'+cols_inputs[i]] #name column
            df_stdcorr = pd.concat([df_stdcorr, df_std], axis=1) #combine multiple correlations

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
    #ax.barh(y_pos, spearman_results['correlation'], bar_width,color='#1f77b4',label='Spearman'+cols_outputs[0])
    # ax.barh(y_pos+bar_width, pearsonr_list, bar_width,
    #         color='#ff7f0e',
    #         label='Pearson')

    #spearman_results = pd.DataFrame(pd.Series(df_corr, cols_inputs))
    #df_corr.iloc[:,0].plot.barh(stacked=True)

    ## PLOT Correlations ##
    plot_corr = 0
    if plot_corr == 1:
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
    plot_heatmap = 0
    if plot_heatmap == 1:
        df_corr = df_corr[abs(df_corr[cols_outputs[0]]) > 0.2]
        heatmap = df_corr.as_matrix(columns=cols_outputs)
        #print(heatmap)
        #cm = plt.get_cmap('spectral')
        fig, ax = plt.subplots()
        im = ax.matshow(heatmap, cmap=cm.Spectral_r, interpolation='none')
        ind = np.arange(len(df_corr))

        ax.set_yticks(ind)  # set positions for y-labels, .5 to put the labels in the middle
        ax.set_yticklabels(df_corr.index)  # set y labels ('variables') from index
        ax.set_yticks(ind + .5, minor=True)
        ax.set_xticklabels('')
        ax.set_xticks(ind)  # set positions for y-labels, .5 to put the labels in the middle
        ax.set_xticklabels(cols_outputs, rotation=90)  # set y labels ('variables') from index
        ax.set_xticks(ind + .5, minor=True)
        ax.grid(which='minor', linewidth=2)
        ax.grid(False, which='major')

        # annotating the data inside the heatmap
        print('print heatmpa data', heatmap[0][0])
        for y in range(df_corr.shape[0]):
            for x in range(df_corr.shape[1]):
                plt.text(x, y, '%.2f' % heatmap[y][x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=9)

        cbar = plt.colorbar(im, fraction=0.04555, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        #ax.legend(bbox_to_anchor=(1, .5), loc='center left')  # bbox_to_anchor=(all the way next to the plot =1 if center left = loc, height 0.5 is in the middle)
        plt.tight_layout(rect=[0, 0, 0.8, 1], pad=.75, w_pad=0.1, h_pad=0.1)  # use rect to adjust the plot sides rects=[left, bottom, right, top]

def surrogate_model(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, TIME_STEP, write_model, write_data):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']
    print(X_train.shape, Y_train.shape)

    lr = LinearRegression()
    lasso = Lasso()
    rr = Ridge()
    pls = PLSRegression() # is really inaccurate, can I increase its accuracy??
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
    fig = plt.figure()
    ax1 = plt.subplot(111)
    y_pos = np.arange(len(models)) # no. of models/methods i am using.
    r2_list, mse_list, time_list, mae_list, evar_list = [], [], [], [], [] # fill lists with evaluation scores

    predictions = pd.DataFrame()
    for i in range(len(cols_outputs)):
        true_series = pd.Series(Y_test[:,i], name=cols_outputs[i]+'_true')
        predictions = pd.concat([predictions, true_series], axis=1)
    model_names = []

    print('columns', cols_outputs)
    ##
    for name, model in models:
        model_names.append(name)

        if model in {bayesm, svrm, gpr}: # use multioutputregressor for these methods
            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            #                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            # surrogate = GridSearchCV(svrm, tuned_parameters)
            multi_model = MultiOutputRegressor(model)

        stime = time.time()

        ## Fitting ##
        # TODO at the moment the linear regression is the most accurate, but its likely others should be as accurate...
        # TODO check also different kernels
        # TODO the surrogate model is built for predicting monthly energy use, what about daily/hourly???
        #print('x-test', X_test)
        #print(X_test.shape)
        if model in {bayesm, svrm, gpr}:
            print(name)
            multi_model.fit(X_train, Y_train)
            prediction = multi_model.predict(X_test)

            if write_model == True:
                joblib.dump(multi_model, DataPath_model_real+name+'_model.pkl')
            else:
                print('write_model set to:', write_model)

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
                # print(prediction[:,1])
                # print('prediction', prediction)
                # print(prediction.shape)
                # print('Y-test', Y_test)
                # print(Y_test.shape)

            if write_model == True:
                joblib.dump(model, DataPath_model_real+name+'_model.pkl')
            else:
                print('write_model set to:', write_model)


        # timing / scoring
        mse_day, mae_day = [], []

        raw_mse = mean_squared_error(Y_test, prediction, multioutput='raw_values')
        raw_mae = mean_absolute_error(Y_test, prediction, multioutput='raw_values')  # multioutput='raw_values' will give the error for all 31days*8enduses (248, )

        print('raw_mse', raw_mse.shape)
        for i in range(8):
            mse_day.append(np.sum(raw_mse[i::8])) # sum mse of all 31 days for every end-use.
            mae_day.append(np.sum(raw_mae[i::8]))
        total_mae = np.sum(mae_day)

        print('mse_day', mse_day)
        print('mae_day', mae_day)
        print('total_mae', total_mae)
        print(len(mse_day))

        time_list.append("{0:.4f}".format(((time.time() - stime))))
        r2_list.append("{0:.4f}".format((r2_score(Y_test, prediction, multioutput='uniform_average'))))
        mse_list.append("{0:.4f}".format((mean_squared_error(Y_test, prediction))))
        mae_list.append("{0:.4f}".format(total_mae))
        evar_list.append("{0:.4f}".format((explained_variance_score(Y_test, prediction))))

        print(len(prediction))
        print('mae', mean_absolute_error(prediction, Y_test))

        #raw_mse = mean_squared_error(Y_test, model.predict(X_test), multioutput='raw_values')
        #print("{0:.4f}".format((mean_squared_error(Y_test, prediction))))

        abs_diff = [(j - i)**2 for i, j in zip(Y_test, prediction)]
        mse_abs_diff = [np.mean(i)/len(Y_test) for i in abs_diff] # the mean squared error for each iteration separately
        abs_perc_error = [abs((i - j)/i)*100 if i > 0 else 0 for i, j in zip(Y_test[1,:], prediction[1,:])]
        abs_perc_error_enduse = []
        for i in range(8):
            abs_perc_error_enduse.append(np.mean(abs_perc_error[i::8])) # sum mse of all 31 days for every end-use.

        print('single prediction', prediction[1,:].tolist())
        print('single test data', Y_test[1,:].tolist())
        print('single abs perc error per day per end use', abs_perc_error)
        print('mean abs perc error per end use', abs_perc_error_enduse)
        #print(len(abs_perc_error))
        #print(mse_abs_diff)
        #print(len(abs_diff[0]))

        raw_rmse = []
        for mse in raw_mse:
            raw_rmse.append(np.sqrt(mse)*1000)

        # plot mean square error for each model
        #df_mse = pd.DataFrame(raw_rmse)
        #df_mse.plot()

        #print(name)


        # Get coefficients and intercepts, this actually works different for each algorithm, so exporting the model is more efficient.
        if model in {bayesm, svrm, gpr}:
            print('model score', models[x][1].score)

            if name == 'SVR':
                df_coef = pd.DataFrame(models[x][1].coef_, cols_outputs)
                df_intercept = pd.DataFrame(models[x][1].intercept_, cols_outputs)
                #print('params', models[x][1].get_params)
        elif model in {lr, rr, nn, knn, rf, lasso, nn, pls}:
            df_coef = pd.DataFrame(models[x][1].coef_, cols_outputs)
            df_intercept = pd.DataFrame(models[x][1].intercept_, cols_outputs)
            #print('coef_', models[x][1].coef_)
            #print('shape coef_', models[x][1].coef_.shape) # for each output Y there and input variables X, there are X by Y number of coefficients.
            #print('intercept_', models[x][1].intercept_) # and for each output Y there are Y number of intercepts

        #print(df_coef.head())
        
        ## Send function coefficients and intercept for each model to csv ##
        df_coef.to_csv(DataPath_model_real + name +'_coef_' + TIME_STEP + '.csv')
        df_intercept.to_csv(DataPath_model_real + name + '_intercept_' + TIME_STEP + '.csv')


        ## Scatterplot of the test data and prediction data for one single iteration
        #TODO combine the diff prediciont from multiple algorithms
        # pred_series = pd.DataFrame(prediction[1, :])
        # test_series = pd.DataFrame(Y_test[1, :])
        # df_pred_test = pd.concat([pred_series, test_series], axis=1)
        # df_pred_test.columns = ['prediction', 'test']
        # print(df_pred_test)
        # df_pred_test.plot(kind='scatter', x='prediction', y='test')

        ## Plotting ##
        for i in range(len(cols_outputs[:8])): # at the moment only 8 days are taken here.
            prediction_series = pd.Series(prediction[:,i], name=cols_outputs[i]+'_'+name)
            predictions = pd.concat([predictions, prediction_series], axis=1)


            prediction_std = np.std(predictions)
            #print(cols_outputs[i], name, prediction[:5], Y_test[:5])
            ax1.plot(prediction[:,i], color=colors[x], label=name if i == 0 else "") # plot first column, and label once

        if write_data == True:
            predictions.to_csv(DataPath_model_real + name +'_predictions_' + TIME_STEP + '.csv')
        x+=1

    #print(Y_test.shape)
    for i in range(len(cols_outputs[:10])):
        ax1.plot(Y_test[:, i], 'o', color=colors[i], label=cols_outputs[i])

    #Y_test = Y_test[Y_test[:,0].argsort()] # sort test data based on frist column
    bar_width = .3

    #print('time [s]:', time_list)
    df_scores = pd.DataFrame(np.column_stack([[x[0] for x in models], mae_list, mse_list]),
                             columns=['Algorithms','Mean absolute error', 'Mean squared error $\mathregular{[kWh/m^{2}a]}$'])
    df_r2 = pd.DataFrame(np.column_stack([[x[0] for x in models], r2_list]), columns=['Algorithms', '$\mathregular{r^{2}}$'])
    df_scores.set_index('Algorithms', inplace=True)
    df_r2.set_index('Algorithms', inplace=True)
    df_scores=df_scores.astype(float) #change data back to floats..
    df_r2 = df_r2.astype(float)
    print(df_scores.head())
    print(df_r2)
    ax = df_scores.plot(kind='bar', stacked=False, width=.3, position=0)
    ax2 = ax.twinx()
    df_r2.plot(ax=ax2, kind='bar', width=.15, position=1, color=colors[5])

    print('learning times [seconds]', time_list)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*.8, box.height])

    ax1.legend(bbox_to_anchor=(1,.5), loc='center left')    #http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax1.set_ylabel('[kWh/m2a]')
    #plt.tight_layout()

    return r2_list


# Run the surrogate learning process with different sample sizes to see the learning rate.
def surrogate_learning_rate():
    df_r2 = pd.DataFrame()
    its = np.arange(20, len(Y_train), 20)
    for i in range(1, len(Y_train)):
        if i % 20 == 0:
            print(i)

            #print(X_train)

            r2_list = surrogate_model(X_train[:i], X_test, Y_train[:i], Y_test, cols_outputs, cols_inputs, write_model=False, write_data=False)
            #print(df_scores)
            r2_list = pd.DataFrame(pd.Series(r2_list))
            df_r2 = pd.concat([df_r2, r2_list])

    df_r2.index = its
    print(df_r2)


def main():


    # Import data from parallel simulations.
    Y_real, cols_outputs, run_numbers = import_outputs(DataPath_model_real, TIME_STEP, NO_ITERATIONS)  # real output data
    # Import data from the spreadsheet that contains all input data.
    X_real, cols_inputs = import_inputs(DataPath_model_real, run_numbers)  # real input data

    # print(X_real[:5])
    # print(Y_real[:5])

    # Cut down data
    # Y_real = Y_real[:50]
    # X_real = X_real[:50]
    print(X_real.shape, Y_real.shape)

    # Test data for testing algorithms.
    # Y_real, X_real, cols_outputs, cols_inputs = test_data(DataPath) # test data

    X_real_data = np.vstack((cols_inputs, X_real))
    Y_real_data = np.vstack((cols_outputs, Y_real))
    X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real)  # randomly split data

    #building the surrogate model using training data
    surrogate_model(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, TIME_STEP, write_model=True, write_data=False)


    # visualise_outputs(Y_real, cols_outputs)


    correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, TIME_STEP)

    # surrogate_learning_rate()



    plt.show()

    #TODO need to rearrange Y_real data so that days are consecutive instead of end-uses, will this change the prediction results???
    input_outputs = np.hstack((X_real_data, Y_real_data))
    with open(DataPath_model_real+ 'input_outputs_' + TIME_STEP + '.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(input_outputs)

if __name__ == '__main__':
    UserName = getpass.getuser()
    DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'
    DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/Run1000/'

    FLOOR_AREA = 5876  # model floor area?
    NO_ITERATIONS = 10
    TIME_STEP = 'month'  # 'year', 'month', 'day', 'hour', 'half-hour' #todo only year and month work atm.

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

