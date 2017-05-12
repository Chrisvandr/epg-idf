import numpy as np
from numpy import linspace
import getpass
import time
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import csv
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

#Sensitivity modules
from SALib.sample import saltelli
from SALib.analyze import sobol, delta, fast, morris, dgsm, ff
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

UserName = getpass.getuser()
DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'

#DataPath_model_real = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/01_CentralHouse/Model/LegionRuns/'
DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/LegionRuns/'

floor_area = 5876 #model floor area?

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
    plt.style.use('ggplot')
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

def import_outputs(DataPath_model_real):
    runs = pd.DataFrame()
    run_no = []
    done = False
    for i in range(300):
        #print(str(format(i, '04')))
        file_name = 'eplusmtr'+str(format(i, '04'))+'.csv'
        rng = pd.date_range(start='1/1/2017 01:00:00', end='2/1/2017',freq='H')

        #TODO if using actual weather file in comparison to real data, i have to set exact months isntead in rng.
        df = pd.read_csv(os.path.join(DataPath_model_real, file_name), header=0, index_col=0)
        df = df[48:] #exclude 2 design days
        df = df.set_index(rng)
        df.index = df.index - pd.DateOffset(hours=1) # !! set all hours back one, so that 01:00 = 00:00 and 24:00 becomes 23:00.
        # So the energy use in the first hour is set at 00:00. This is in line with the submetered data
        # Also, pandas works from 0-23hours, it would otherwise count 24 towards the next day.

        plot_run = 1 # plotting single runs.
        if plot_run == 0:
            if i == 20: #plot which run
                df = df[:24*7] #length 7 days
                df = df.div(3600*1000*floor_area)
                PlotAllArea(df)
                done = True
                break

        df_year = df.resample('A').sum()
        run_no.append(file_name[-8:-4]) # get run number
        runs = pd.concat([runs, df_year], axis=0)

        if done == True:
            sys.exit('plotting single run and breaking')
            break

    cols = runs.columns.tolist()
    cols_outputs = []
    run_no = pd.DataFrame(pd.Series(run_no)) # create pandas dataframe of run numbers
    #print(run_no)
    run_no.columns = ['run_no']
    run_no = run_no.ix[:, 0].str.replace("_", "0") # TODO for now replace run no undersscores
    for i, v in enumerate(cols):
        v = v[:10]
        cols_outputs.append(v)
    runs.reset_index(drop=True, inplace=True) # throw out the index (years)
    runs.columns = cols_outputs  # rename columns
    runs = pd.concat([run_no, runs], axis=1, ignore_index=False) # prepend the run numbers to existing dataframe

    ## Output the energy predictions to .csv ##
    runs.to_csv(DataPath_model_real + 'runs_outputs.csv')
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

    runs = runs.div(3600*1000*floor_area) # convert Joules to kWh

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

    return Y_real, cols_outputs

def import_inputs(DataPath_model):
    df = pd.read_csv(DataPath_model_real + 'inputs_CentralHouse_222backup2.csv', header=0) #TODO using backup file atm
    cols_inputs = df.columns.tolist()
    #print(["%.2d" % x for x in (np.arange(0, df.shape[0]))]) #zero pad a list of numbers
    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)
    X_real = df.ix[:,:]
    X_real = X_real.drop(df.columns[[0]], axis=1) #drop run_no column
    print('X_real shape', X_real.shape)
    X_real = X_real.as_matrix()

    return X_real, cols_inputs

Y_real, cols_outputs = import_outputs(DataPath_model_real) # real output data
X_real, cols_inputs = import_inputs(DataPath_model_real) # real input data



#Cut down data
#Y_real = Y_real[:50]
#X_real = X_real[:50]
print(X_real.shape, Y_real.shape)
#TEST DATA
#Y_real, X_real, cols_outputs, cols_inputs = test_data(DataPath) # test data

X_real_data = np.vstack((cols_inputs, X_real))
Y_real_data = np.vstack((cols_outputs,Y_real))
X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real) #randomly split data

#put inputs and outputs in one csv file for review.
input_outputs = np.hstack((X_real_data, Y_real_data))
with open(DataPath_model_real+"input_outputs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(input_outputs)

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
#visualise_outputs(Y_real, cols_outputs)

#Correlation coefficients on real data

def correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs):
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
    df_corr.to_csv(DataPath_model_real + 'correlations.csv')
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

    plt.show()
#correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs)



def surrogate_model(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    lr = LinearRegression()
    lasso = Lasso()
    rr = Ridge()
    pls = PLSRegression() # is really inaccurate, can I increase its accuracy??
    knn = KNeighborsRegressor(5, weights='uniform')
    nn = MLPRegressor(hidden_layer_sizes=(100,), solver='lbfgs')
    rf = RandomForestRegressor()

    #ransac = RANSACRegressor()
    #hr = HuberRegressor()

    print(X_train.shape, Y_train.shape)

    # do not support multivariate regression, use MultiOutputRegressor
    bayesm = BayesianRidge()
    svrm = SVR(kernel='linear', C=1000) # takes about 100s
    gpr = GaussianProcessRegressor(C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))

    #("PLS", pls), ("SVR", svrm), # take too long
    #, ("rr", rr),  ("k-NN", knn), ("NN", nn), ("RF", rf), ("BR", bayesm), ("Lasso", lasso), ("NN", nn), ("GPR", gpr)
    models = [("LR", lr), ("rr", rr)] # is of type list #print(type(models))

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

    ##
    for name, model in models:
        model_names.append(name)
        if model in {bayesm, svrm, gpr}: # use multioutputregressor for these methods
            # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            #                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            # surrogate = GridSearchCV(svrm, tuned_parameters)
            multi_model = MultiOutputRegressor(model)

        stime = time.time()

        # fitting
        if model in {bayesm, svrm, gpr}:
            prediction = multi_model.fit(X_train, Y_train).predict(X_test)
        elif model in {lr, rr, nn, knn, rf, lasso, nn, pls}:
            prediction = model.fit(X_train, Y_train).predict(X_test)
        #print(prediction[:,1])

        # timing / scoring
        time_list.append("{0:.4f}".format(((time.time() - stime))))
        r2_list.append("{0:.4f}".format((r2_score(Y_test, prediction, multioutput='uniform_average'))))
        mse_list.append("{0:.4f}".format((mean_squared_error(Y_test, prediction))))
        mae_list.append("{0:.4f}".format((mean_absolute_error(Y_test, prediction))))
        evar_list.append("{0:.4f}".format((explained_variance_score(Y_test, prediction))))
        #print "r2:", model.score(X_test, Y_test)

        print(name)
        if model in {bayesm, svrm, gpr}:
            print('coef_', models[x][1].score)
            #print('params', models[x][1].get_params)
        elif model in {lr, rr, nn, knn, rf, lasso, nn, pls}:
            df_coef = pd.DataFrame(models[x][1].coef_, cols_outputs)
            df_intercept = pd.DataFrame(models[x][1].intercept_, cols_outputs)
            #print('coef_', models[x][1].coef_)
            #print('shape coef_', models[x][1].coef_.shape) # for each output Y there and input variables X, there are X by Y number of coefficients.
            #print('intercept_', models[x][1].intercept_) # and for each output Y there are Y number of intercepts

        #print(df_coef.head())
        ## Send function coefficients and intercept for each model to csv ##
        df_coef.to_csv(DataPath_model_real + name +'_coef.csv')
        df_intercept.to_csv(DataPath_model_real + name + '_intercept.csv')

        ## Plotting ##
        for i in range(len(cols_outputs)):
            prediction_series = pd.Series(prediction[:,i], name=cols_outputs[i]+'_'+name)
            predictions = pd.concat([predictions, prediction_series], axis=1)

            #print(cols_outputs[i], name, prediction[:5], Y_test[:5])
            ax1.plot(prediction[:,i], color=colors[x], label=name if i == 0 else "") # plot first column, and label once
        x+=1


    #print(X_test[0])
    #x0 = np.array(X_test[0])  # single run of test_data
    #print(x0)
    #print(df_coef)
    #f(x0, df_coef, df_intercept)
    #x, cov, infodict, mesg, ier = minimize(f, x0[:], args=(df_coef, df_intercept), method='SLSQP')
    #print('calibrated variables', x)


    for i in range(len(cols_outputs)):
        ax1.plot(Y_test[:, i], 'o', color=colors[i], label=cols_outputs[i])

    #print(predictions[:5])
    predictions.to_csv(DataPath_model_real+'predictions.csv')

    #Y_test = Y_test[Y_test[:,0].argsort()] # sort test data based on frist column
    bar_width = .3

    #print('time [s]:', time_list)
    df_scores = pd.DataFrame(np.column_stack([[x[0] for x in models], r2_list, mae_list, mse_list, time_list]),
                             columns=['Algorithms','$\mathregular{r^{2}}$', 'Mean absolute error', '$\mathregular{Mean squared error [kWh/m^{2}a]}$', 'Training time [s]'])
    df_scores.set_index('Algorithms', inplace=True)
    df_scores=df_scores.astype(float) #change data back to floats..
    df_scores.plot.bar(stacked=False)
    #print(df_scores)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*.8, box.height])

    ax1.legend(bbox_to_anchor=(1,.5), loc='center left')    #http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax1.set_ylabel('[kWh/m2a]')
    #plt.tight_layout()
    plt.show()
#surrogate_model(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs)



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

