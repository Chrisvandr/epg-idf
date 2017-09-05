import numpy as np
from numpy import linspace
import getpass
import time
import pandas as pd
import os
import sys
import csv
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

UserName = getpass.getuser()
#own scripts
sys.path.append('C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/UCLDataScripts')
from PlotSubmetering import ReadRuns
from PlotSubmetering import lineno


def import_outputs(DataPath_model_real, time_step, NO_ITERATIONS, run_numbers):
    runs = pd.DataFrame()
    done = False

    list_iterations = []
    for i in range(NO_ITERATIONS):
        list_iterations.append(i)

    print('importing data at TIME_STEP:', time_step)
    data_runs = []

    def strip_columns(df):
        cols = df.columns.tolist()
        for i, v in enumerate(cols):  # rename columns
            if 'THERMAL ZONE' in cols[i]:
                rest = cols[i].split(':', 1)[1]
                rest = ''.join(cols[i].partition('ZONE:')[-1:])
                rest = re.sub("([(\[]).*?([)\]])", "\g<1>\g<2>", rest)  # remove text within symbols
                rest = re.sub("[(\[].*?[)\]]", "", rest)  # remove symbols
                rest = rest.strip()  # remove leading and trailing spaces
                rest = rest.lower()
            elif ':' not in cols[i]:
                rest = cols[i]
                rest = re.sub("([(\[]).*?([)\]])", "\g<1>\g<2>", rest)  # remove text within symbols
                rest = re.sub("[(\[].*?[)\]]", "", rest)  # remove symbols
                rest = rest.strip()  # remove leading and trailing spaces
                rest = rest.lower()

            else:
                rest = cols[i].split(':', 1)[0]  # removes all characters after ':'
                rest = rest.replace('Interior', '')
            # print(rest)
            df.rename(columns={cols[i]: rest}, inplace=True)
        return df

    def read_eplusout(df):
        if building_abr in {'CH', 'MPEB'}:

            if df.shape[0] < 10000:  # hourly
                df = df[48:]  # exclude 2 design days
                rng = pd.date_range(start='01/01/2017 01:00:00', end='01/01/2018',
                                    freq='H')  # rng = pd.date_range(start='01/01/2017 01:00:00', end='01/01/2018', freq='15Min')
                df = df.set_index(rng)
                df.index = df.index - pd.DateOffset(
                    hours=1)  # !! set all hours back one, so that 01:00 = 00:00 and 24:00 becomes 23:00.

                # separate out the df in two pieces, then rename the second part as 2017 data in order to align with measurements
                df_first = df.loc['2017-09-01':'2018-01-01']
                df_second = df.loc['2017-01-01':'2017-04-30']
                rng = pd.date_range(start='2016-09-01', end='2016-12-31 23:00:00', freq='H')
                df_first = df_first.set_index(rng)

            else:  # 15Min
                df = df[48 * 4:]  # exclude 2 design days
                rng = pd.date_range(start='01/01/2017 00:15:00', end='01/01/2018',
                                    freq='15Min')  # rng = pd.date_range(start='01/01/2017 01:00:00', end='01/01/2018', freq='15Min')
                df = df.set_index(rng)
                df.index = df.index - pd.DateOffset(
                    hours=.25)  # !! set all hours back one, so that 01:00 = 00:00 and 24:00 becomes 23:00.
                # So the energy use in the first hour is set at 00:00. This is in line witfh the submetered data
                # Also, pandas works from 0-23 hours, it would otherwise count 24 towards the next day.

                # separate out the df in two pieces, then rename the second part as 2017 data in order to align with measurements
                df_first = df.loc['2017-09-01':'2018-01-01']
                df_second = df.loc['2017-01-01':'2017-04-30']
                rng = pd.date_range(start='2016-09-01', end='2016-12-31 23:45:00', freq='15Min')
                df_first = df_first.set_index(rng)

            df = pd.concat([df_first, df_second], axis=0)
            df = strip_columns(df)

    for i in run_numbers:
        #print(str(format(i, '04')))
        if i % 50 == 0:
            print(i)

        file_name = 'eplusmtr'+str(format(i, '04'))+'.csv'
        if os.path.isfile(os.path.join(DataPath_model_real, file_name)):
            df = pd.read_csv(os.path.join(parallel_runs_harddisk, file_name), header=0, index_col=0)
            df = read_eplusout(df)

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
                    df = df.div(3600*1000*floor_area)
                    PlotAllArea(df)
                    done = True
                    break

            # train the surrogate models based on predicted data that is equal to the measured data. (no need to include all data)
            if building_abr == 'CH':
                # !! IMPORTANT, the surrogate model is trained to predict only the months that are measured !!
                df = df[(df.index.month <= 4) | (df.index.month >= 9)]  # exclude first 4 months
                #print(df.index.month.tolist())
                df_systems = pd.concat([df.ix[:, 0], df.ix[:, 4:6], df.ix[:, 7]], axis=1).sum(axis=1)
                df_landp = pd.concat([df.ix[:, 1:4]], axis=1).sum(axis=1)
                df_gas = pd.concat([df.ix[:, 6]], axis=1).sum(axis=1)
                df = pd.concat([df_systems, df_landp, df_gas], axis=1)
                df.columns = ['Systems', 'L&P', 'Gas']

            if building_abr == 'MPEB':
                df_pred = df

                # Separate PLant and FCU fans
                df_pred_plant_fans = df_pred[['ref1', 'ref2', 'ref4a',
                                              'ref4b', 'ref5', 'ahu1_extract', 'ahu1_supply', 'ahu2_extract',
                                              'ahu2_supply', 'ahu3_supply', 'ahu4_supply', 'ahu5_extract',
                                              'ahu5_supply']]
                df_pred_plant_fans = df_pred_plant_fans.sum(axis=1)
                df_pred_plant_fans = pd.DataFrame(df_pred_plant_fans)
                df_pred_plant_fans.rename(columns={df_pred_plant_fans.columns[0]: 'Plant Fans'}, inplace=True)
                df_pred_fans = pd.concat([df_pred_plant_fans, df_pred[['Fans']]], axis=1)
                df_pred_fans['FCU Fans'] = df_pred_fans['Fans'] - df_pred_fans['Plant Fans']

                # Separate Servers and Equipment
                df_pred_servers = df_pred[['g01b machine room', '409 machine room']].sum(axis=1)
                df_pred_servers = pd.DataFrame(df_pred_servers)
                df_pred_servers.rename(columns={df_pred_servers.columns[0]: 'Servers'}, inplace=True)
                df_pred_equipment = pd.concat([df_pred['Equipment'], df_pred_servers], axis=1)
                df_pred_equipment['Equipment'] = df_pred['Equipment'] - df_pred_equipment['Servers']

                df = pd.concat([df_pred[['Pumps', 'Heating', 'Cooling']], df_pred_fans[['Plant Fans']],
                                df_pred[['Lights', 'WaterSystems']], df_pred_equipment[['Equipment']],
                                df_pred_fans[['FCU Fans']], df_pred_servers], axis=1)

            if time_step == 'year': # with a full year of simulation, 1year*8enduses
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

            elif time_step == 'month':
                df = df.resample('M').sum()
                df = df[(df.index.month <= 4) | (df.index.month >= 9)]  # exclude first 4 months again
                cols = df.columns.tolist()

                if END_USES == False:
                    df_month = df.sum(axis=1)
                    df_month = pd.DataFrame(df_month)
                    runs = pd.concat([runs, df_month.T], axis=0)

                elif END_USES == True:
                    df_month = list(itertools.chain.from_iterable(df.values.tolist()))
                    data_runs.append(df_month)

                if i == list_iterations[-1]:
                    if END_USES == False:
                        df = df.sum(axis=1)
                        df_index = df.index.tolist()
                        print(df_index)
                        months_from_index = []
                        for month in df_index:
                            months_from_index.append(str(month.month)) #add month number to column
                        runs.columns = months_from_index
                        #runs.reset_index(drop=True, inplace=True)

                    if END_USES == True:
                        df_index = df.index.tolist()
                        months_from_index = []
                        for month in df_index:
                            for col in cols:
                                months_from_index.append(str(month.month)+'_'+str(col)) #add month number to column
                        runs = pd.DataFrame(data_runs, columns = months_from_index)
                        runs.reset_index(drop=True, inplace=True)

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
                    print(runs)


            elif time_step == 'day': # with a full year of simulation, 265days*8enduses
                df_day = df.resample('D').sum()
                cols = df_day.columns.tolist()
                days = []

                # transpose days per objective, have the days on the x-axis
                for col in cols:
                    days.extend(df_day[col].tolist())

                df_days = pd.DataFrame(pd.Series(days))
                runs = pd.concat([runs, df_days.T], axis=0) #todo this is probably not correct, check month

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

            elif time_step == 'hour':
                df_hour = df.resample('H').sum()
                runs = pd.concat([runs, df_hour], axis=0)

            if done == True:
                sys.exit('plotting single run and breaking')
                break

    # Set run numbers as index
    run_no = []
    run_no = np.array(run_numbers) # make array of list
    runs['run_no'] = run_no # append array to dataframe
    cols = runs.columns.tolist()
    cols = cols[-1:] + cols[:-1] # rearrange columns, put runs_no as first column
    runs = runs[cols]

    ## Output the energy predictions to .csv ##
    runs.to_csv(DataPath_model_real + 'runs_outputs_'+time_step+'.csv')
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


    cols_outputs = runs.columns.tolist()
    Y_real = runs.as_matrix()


    return Y_real, cols_outputs

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

            kde = gaussian_kde(df[v])
            dist_space = linspace(min(df[v]), max(df[v]), 100)
            #print(dist_space)
            plt.plot(dist_space, kde(dist_space), label=v, color=COLORS[i])

    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.tight_layout()
    plt.legend(loc='best')
    plt.ylabel('Distribution')
    plt.xlabel('Energy [kWh/m2a]')
    plt.show()

def MultiBarPlot(df):
    if len(df.shape) == 1: # if df only exists of one variable/columns
        df.plot(kind='bar', stacked=True,legend='best', color='#1f77b4', alpha=.5)
    elif len(df.shape)>1: # for multiple columns

        cols = df.columns.tolist()
        df.plot(kind='bar', stacked=True,legend='best', color=COLORS, alpha=.5)

    plt.gcf().autofmt_xdate()
    plt.ylabel('Energy (kWh)')
    plt.show()
def MultiBarBoxPlot(df):
    if len(df.shape) == 1: # if df only exists of one variable/columns
        df.plot.box()
    elif len(df.shape)>1: # for multiple columns

        cols = df.columns.tolist()
        df.plot.box()

    plt.legend(loc='best')
    plt.ylabel('Energy (kWh)')
    plt.show()

def plot_surrogate_learning_rate():
    df_r2 = pd.DataFrame()
    df_mse = pd.DataFrame()
    its = np.arange(20, len(Y_train), 20)
    for i in range(1, len(Y_train)):
        if i % 20 == 0:
            print(i)
            r2_list, mse_list = surrogate_model(X_train[:i], X_test, Y_train[:i], Y_test, cols_outputs, cols_inputs,
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

def correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step):
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
    df_corr.to_csv(DataPath_model_real + 'correlations' + time_step + '.csv')
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
    # ax.barh(y_pos, spearman_results['correlation'], bar_width,color='#1f77b4',label='Spearman'+cols_outputs[0])
    # ax.barh(y_pos+bar_width, pearsonr_list, bar_width,
    #         color='#ff7f0e',
    #         label='Pearson')

    #spearman_results = pd.DataFrame(pd.Series(df_corr, cols_inputs))
    #df_corr.iloc[:,0].plot.barh(stacked=True)
    return df_corr

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
def plot_heatmap_correlations(df, cols_outputs):
    #df_corr = df_corr[abs(df_corr[cols_outputs[0]]) > 0.3] # based on first columns
    #df_corr = df_corr[df_corr > .5]

    print(df)
    #df = df[df[df > 10].any(1)] # select rows where any column is greater than 10
    #df = df[abs(df[abs(df) > .25].count(axis=1) > 0.35)] # where any value is greater than .5 and any column is greater than 1
    #todo perhaps exclude the server_exclusive input as it is the strongest correlator...)
    df = df[abs(df['Cooling'] > .25)]
    #vif_indices = [i for i, v in enumerate(vif) if v == np.inf or v < 6]  # when to exclude features

    heatmap = df.as_matrix(columns=cols_outputs)
    print(heatmap)
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

    ax.grid(which='minor', linewidth=2, color='white')
    ax.grid(False, which='major')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # annotating the data inside the heatmap
    print('print heatmpa data', heatmap[0][0])
    for y in range(df.shape[0]):
        for x in range(df.shape[1]):
            plt.text(x, y, '%.2f' % heatmap[y][x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=8)

    cbar = plt.colorbar(im, fraction=0.04555, pad=0.04)
    cbar.ax.tick_params()
    #ax.legend(bbox_to_anchor=(1, .5), loc='center left')  # bbox_to_anchor=(all the way next to the plot =1 if center left = loc, height 0.5 is in the middle)
    plt.tight_layout(rect=[0, 0, 0.8, 1], pad=.75, w_pad=0.1, h_pad=0.1)  # use rect to adjust the plot sides rects=[left, bottom, right, top]

def import_inputs(DataPath_model_real, inputs, run_numbers):
    df = pd.read_csv(DataPath_model_real + inputs, header=0) #TODO using backup file atm
    cols_inputs = df.columns.tolist()

    runs_no_inputs = pd.DataFrame(["%.2d" % x for x in (np.arange(0, df.shape[0]))])
    df = pd.concat([runs_no_inputs, df], axis=1, ignore_index=True)
    X_real = df.ix[run_numbers]

    X_real = X_real.drop(df.columns[[0]], axis=1) #drop run_no column
    print(lineno(), X_real.head(2))
    X_real = X_real.as_matrix()

    # feature selection

    #  univariate selection
    univariate_selection = True
    if univariate_selection is True:
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=(.8 * (1 - 0.8)))
        sel.fit(X_real) # fit the inputs
        selected_feature_indices = sel.get_support(True) #get indices of those selected
        print(lineno(), 'selected features', len(selected_feature_indices), selected_feature_indices)
        X_real = sel.transform(X_real) #transform the matrix
        print(lineno(), X_real)
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
def surrogate_model(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, plot_progression, write_model, write_data):
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

def main():
    # Import data from parallel simulations and real data.
    df_pred, runs, df_runs = ReadRuns(building, base_case, floor_area, building_harddisk, building_abr, time_step, write_data, NO_ITERATIONS)

    cols_outputs = runs.columns.tolist()
    #print(lineno(), runs)
    Y_real = runs.as_matrix()

    print(lineno(), Y_real[:2])

    X_real, cols_inputs = import_inputs(DataPath_model_real, inputs, run_numbers)  # real input data

    # Cut down data
    # Y_real = Y_real[:50]
    # X_real = X_real[:50]
    print(lineno(), X_real.shape, Y_real.shape)

    # Test data for testing algorithms.
    # Y_real, X_real, cols_outputs, cols_inputs = test_data(DataPath) # test data

    X_real_data = np.vstack((cols_inputs, X_real))
    Y_real_data = np.vstack((cols_outputs, Y_real))
    X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real)  # randomly split data to create test set

    #building the surrogate model using training data
    #surrogate_model(X_train, X_test, Y_train, Y_test, cols_outputs, cols_inputs, time_step, plot_progression, write_model, write_data)



    df_corr = correlations(DataPath_model_real, X_real, Y_real, cols_outputs, cols_inputs, time_step)
    print(lineno(), df_corr.head())
    #plot_correlations(df_corr, cols_outputs)
    plot_heatmap_correlations(df_corr, cols_outputs)
    # plotting

    # plot_surrogate_learning_rate()

    feature_scores = []
    for i in range(Y_real.shape[1]):
        f_test, _ = f_regression(X_real, Y_real[:,i])
        feature_scores.append(f_test)
        #print('f test', f_test)
    df_feature_scores = pd.DataFrame(list(map(list, zip(*feature_scores))), columns=cols_outputs, index=cols_inputs) # transpose list of lists and make a dataframe of it
    #df_feature_scores.plot(kind='bar', stacked=True)



    input_outputs = np.hstack((X_real_data, Y_real_data))
    if write_data is True:
        with open(DataPath_model_real+ 'input_outputs_' + time_step + '.csv', "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(input_outputs)

    plt.show()

    #TODO need to rearrange Y_real data so that days are consecutive instead of end-uses, will this change the prediction results???


if __name__ == '__main__':
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    BuildingList = ['05_MaletPlaceEngineering', '01_CentralHouse', '02_BuroHappold_17', '03_BuroHappold_71']  # Location of DATA
    BuildingHardDisk = ['05_MaletPlaceEngineering_Project', '01_CentralHouse_Project', '02_BuroHappold_17', '03_BuroHappold_71']
    DataFilePaths = ['MPEB', 'CentralHouse', '17', '71']  # Location of STM data and file naming
    BuildingLabels = ['MPEB', 'Central House', 'Office 17', 'Office 71']
    BuildingAbbreviations = ['MPEB', 'CH', '17', '71', 'Nothing']
    InputVariables = ['inputs_MaletPlace11_08_16_11.csv', 'inputs_CentralHouse_22225_06_11_49.csv']

    FloorAreas = [9579, 5876, 1924, 1691]

    building_num = 0 # 0 = MPEB, 1 = CH, 2 = 17, 3 = 71
    base_case = False # only show a single run from the basecase, or multiple runs (will change both import and plotting)
    NO_ITERATIONS = 1000
    time_step = 'year'  # 'year', 'month', 'day', 'hour', 'half-hour' #todo only year and month work atm.
    end_uses = False # for month specifically (when True it will divide end-uses by month instead of aggregating to total)
    write_model = False # writes out the .pkl model.
    write_data = False # Writes out several .csv files some used for function optimisation script atm.
    plot_progression = False # this will plot the learning progression of the model when set to True.

    building_abr = BuildingAbbreviations[building_num]
    datafile = DataFilePaths[building_num]
    building = BuildingList[building_num]
    building_harddisk = BuildingHardDisk[building_num]
    building_label = BuildingLabels[building_num]
    floor_area = FloorAreas[building_num]
    inputs = InputVariables[building_num]

    DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'
    DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/' + building_harddisk + '/Run1000/'

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

