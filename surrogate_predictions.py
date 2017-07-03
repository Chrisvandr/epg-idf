import numpy as np
import matplotlib.pyplot as plt
import math
import array
import operator
import random
import getpass
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
# from deap import base
# from deap import benchmarks
# from deap import creator
# from deap import tools
# from deap import algorithms
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from sklearn.externals import joblib
#Sensitivity modules
from SALib.sample import saltelli
from SALib.analyze import sobol, delta, fast, morris, dgsm, ff
from SALib.test_functions import Ishigami
from decimal import *

def elements(df_inputs, lower_bound, upper_bound):
    #elements = [random.uniform(a, b) for a, b in zip(lower_bound, upper_bound)]
    elements = []
    for x in range(df_coefs.shape[1]):
            mu, sigma = df_inputs.iloc[0][x], df_inputs.iloc[0][x] * (1 / 10)
            elements.append(random.gauss(mu, sigma)) # set to 10% variation currently... #TODO change
    print([float(Decimal("%.2f" % e)) for e in elements])
    #data = [float(Decimal("%.2f" % e)) for e in mylist]
    #myRoundedList = [round(elem, 2) for elem in myList]
    print([float(Decimal("%.2f" % e)) for e in df_inputs.iloc[0, :df_coefs.shape[1]].tolist()])
    return elements


# Load the pickle surrogate model made by 'surrogate_model.py' and calculate with new inputs
def calculate(individual):
    model = joblib.load(DataPath_model_real + 'rr_model.pkl')
    individual = np.array(individual).reshape((1,-1))
    prediction = model.predict(individual)[0]
    #print(prediction)
    return prediction

## Uses calculation function and creates both dataframes and matrix for inputs/outputs
def surrogate_model(no_runs, df_inputs, lower_bound, upper_bound, cols_outputs, cols_inputs):
    individuals = []
    predictions = []

    for x in range(no_runs):
        individual = elements(df_inputs, lower_bound, upper_bound)
        print('first individual input', individual)
        individuals.append(individual)
        prediction = calculate(individual)
        print(prediction)
        predictions.append(prediction)
        if x % 10 == 0:
            print(x)

    df_predicted = pd.DataFrame(predictions, columns=cols_outputs)
    df_population = pd.DataFrame(individuals, columns=cols_inputs)

    Y_real = df_predicted.as_matrix()
    X_real = df_population.as_matrix()

    print('X_real and Y_real shapes', X_real.shape, Y_real.shape)

    return(X_real, Y_real, df_predicted, df_population)

#Sobol does not work for the test data as it is sampled by LHS and it needs to be sampled by saltelli sampling (which creates many more variables), however it can be done with the surrogate model instead.
def sensitivity_analysis(cols_outputs, cols_inputs):
    # https://media.readthedocs.org/pdf/salib/latest/salib.pdf | http://salib.github.io/SALib/ # http://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    # http://keyboardscientist.weebly.com/blog/sensitivity-analysis-with-salib

    # TODO determine which parameters have a small influence on outputs and use meta-model order reduction by disregarding them. Less samples will be needed for Sobol.
    D = len(cols_inputs) #no. of parameters
    N = 1 #
    second_order = True
    if second_order == True:
        samples = N * (2 * D + 2)
    else:
        samples = N * (D + 2)
    print('no of samples', samples)

    print(D*N, 'for Fast, Delta and DGSM') # (N*D) = [Fast, Delta, DGSM]
    print((D+1)*N, 'for Morris') # (D + 1)*N = [Morris, ]
    print((D+2)*N,'for Sobol first order')
    print(N*(2*D + 2),'for Sobol second order') #, (D+2)*N for first order = [Sobol, ]
    print(2**(round(np.sqrt(150),0)), 'for fractional fast') # (2**D) < N = [FF, ] (so if 300paramaters N = 2^9 > 300)

    ## DUMMY SOBOL ##
    def test_sobol():
        no_vars = 4
        col_names = ['x1', 'x2', 'x3', 'x4']
        problem = {
            'num_vars': no_vars,
            'names': col_names,
            'bounds': [[-np.pi, np.pi]]*no_vars
        }
        # N(2D+2)
        param_values = saltelli.sample(problem, 400, calc_second_order=True)
        print('param_values:', param_values.shape, param_values[:5])

        Y = Ishigami.evaluate(param_values)
        print('Y shape:', Y.shape, Y[:5])

        print(problem)
        Si = sobol.analyze(problem, Y, calc_second_order=True)
        print(Si['ST'])
        print("x1-x2:", Si['S2'][0,1]) #interactive effects between x1 and x2.
    #test_sobol()

    #uniform
    #print([lower_bound], [upper_bound])
    #normal

    real_cols = cols_inputs #columns names of the parameters
    problem = {
        'num_vars': D,
        'names': real_cols,
        'bounds': norm_bounds*D # bound set to the input parameters
    }

    ## Create samples and outputs

    ## TODO have to use Saltelli sampling because otherwise it uses just randomised gauss, and the likely error rates would be higher..
    ## http://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    X_real, Y_real, df_predicted, df_population = surrogate_model(samples, df_inputs, lower_bound, upper_bound, cols_outputs,
                                                                  cols_inputs)
    print('Sobol problem', problem)

    ## Analyze Sobol
    # If calc_second_order is False, the resulting matrix has N * (D + 2) rows, where D is the number of parameters. If calc_second_order is True, the resulting matrix has N * (2D + 2) rows.
    Si_sobol = sobol.analyze(problem, Y_real[:,0], calc_second_order=second_order)
    # TODO sobol is analysed for a single output here. Each output has then x no. of variable correlations.


    #Si_delta = delta.analyze(real_problem, X_real, Y_real[:,0])
    #print(Si_delta['S1'])
    #print(Si_delta['delta'])

    #Si_fast = fast.analyze(real_problem, Y_real[:,0])
    #  Output sample must be multiple of D (parameters)
    #print(Si_fast['S1'])

    #Si_morris = morris.analyze(real_problem, X_real, Y_real[:,0])
    #  Output samplmes must be multiple of D+1 (parameters)
    #print(Si_morris['mu'])

    #Si = dgsm.analyze(real_problem, X_real, Y_real[:,0])
    #print(Si['delta']) # Returns a dictionary with keys ‘delta’, ‘delta_conf’, ‘S1’, and ‘S1_conf’

    #Si_ff = ff.analyze(real_problem, X_real, Y_real[:,0])
    # The resulting matrix has D columns, where D is smallest power of 2 that is greater than the number of parameters.
    # print(Si_ff['ME'])

    fig = plt.figure(0)
    ax = plt.subplot(111)
    bar_width = .45
    y_pos = np.arange(D)
    #print(y_pos)

    ax.bar(y_pos, Si_sobol['S1'], bar_width, label='First-order')
    ax.bar(y_pos+bar_width, Si_sobol['ST'], bar_width, label='Total order')

    print(cols_inputs[0], '-', cols_inputs[1], Si_sobol['S2'][0, 1])

    plt.xticks(y_pos+bar_width / 2, real_cols, rotation='vertical')
    plt.tight_layout()
    plt.legend()

## Plotting the surrogate model predictions and
def objective_plots():
    X_real, Y_real, df_predicted, df_population = surrogate_model(30, df_inputs, lower_bound, upper_bound, cols_outputs, cols_inputs)

    fig, axes = plt.subplots(nrows=5,ncols=5, figsize=(20/2.54, 20/2.54)) #sharex=True, sharey=True

    for i in range(len(targets)):
        dist_space = linspace(min(df_predicted[cols_outputs[i]]), max(df_predicted[cols_outputs[i]]), 50)
        kCOLORS = [COLORS[1] if targets[i]-targets[i]/10 < value < targets[i]+targets[i]/10 else COLORS[0] for value in dist_space]
        for j in range(len(targets)):
            if i == j: # Plot gaussian distribution graphs
                kde = gaussian_kde(df_predicted[cols_outputs[i]])
                axes[i,j].scatter(dist_space, kde(dist_space), c=kCOLORS)

                if j == 0: # add ylabel at first plot 0,0
                    axes[i,j].set_ylabel(cols_outputs[0])
                if j == len(targets)-1: # add xlabel at last plot
                    axes[i, j].set_xlabel(cols_outputs[-1])

                if i != len(targets)-1: # remove labels and ticks for inner plots
                    axes[i, j].get_xaxis().set_ticklabels([])
                if j != 0: # remove labels and ticks for inner plots
                    axes[i, j].get_yaxis().set_ticklabels([])

            else: # Plot scatterplots comparing two objectives
                sCOLORS = [COLORS[1] if targets[i] - (valuex / 10) < valuex < targets[i] + valuex / 10 and
                                    targets[j] - valuey / 10 < valuey < targets[j] + valuey / 10 else
                                    COLORS[0] for valuex, valuey in zip(df_predicted[cols_outputs[i]], df_predicted[cols_outputs[j]])]

                axes[j,i].scatter(df_predicted[cols_outputs[i]], df_predicted[cols_outputs[j]], c=sCOLORS)
                axes[j,i].set_xlabel(cols_outputs[i])
                axes[j,i].set_ylabel(cols_outputs[j])

                if i != len(targets)-1: # remove labels and ticks for inner plots
                    axes[i, j].get_xaxis().label.set_visible(False)
                    axes[i, j].get_xaxis().set_ticklabels([])
                if j != 0: # remove labels and ticks for inner plots
                    axes[i, j].get_yaxis().label.set_visible(False)
                    axes[i, j].get_yaxis().set_ticklabels([])

            #axes[0, j].set_ylabel(cols_outputs[i])
            #axes[i, 0].set_ylabel(cols_outputs[i])
    plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.2)

def main():

    no_runs = 2
    X_real, Y_real, df_predicted, df_population = surrogate_model(no_runs, df_inputs, lower_bound, upper_bound,
                                                                  cols_outputs,
                                                                  cols_inputs)

    # sensitivity_analysis(cols_outputs, cols_inputs)
    # objective_plots()

    plt.show()

if __name__ == '__main__':
    TIME_STEP = 'year'  # 'year', 'month', 'day', 'hour', 'half-hour'
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94',
              '#ff9896', '#c5b0d5', 'red', 'green', 'blue', 'black', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b',
              '#d62728',
              '#9467bd', '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5', 'red', 'green', 'blue',
              'black']

    UserName = getpass.getuser()
    DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'
    DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/Run1000/'

    df_coefs = pd.read_csv(DataPath_model_real + 'LR_coef_' + TIME_STEP + '.csv', header=0, index_col=0)
    df_intercepts = pd.read_csv(DataPath_model_real + 'LR_intercept_' + TIME_STEP + '.csv', header=0, index_col=0)
    df_inputs = pd.read_csv(DataPath_model_real + 'input_outputs_' + TIME_STEP + '.csv', header=0)

    cols = df_inputs.columns.tolist()
    cols_outputs = cols[df_coefs.shape[1]:df_coefs.shape[1] + df_coefs.shape[0]]
    print(cols_outputs)

    targets = [.7, 0.5, 1.5, 1.7, .2]  # , 3.5, 1.2, 0.02]

    mus = df_inputs.iloc[0, :df_coefs.shape[1]]
    sigmas = df_inputs.iloc[0, :df_coefs.shape[1]] * (1 / 10)

    norm_bounds = pd.concat([mus, sigmas], axis=1)
    norm_bounds = norm_bounds.values.tolist()  # turn into list [[mu, sigma], [mu, sigma] ...]

    sigmas = sigmas.tolist()
    print('first line of inputs', df_inputs.iloc[0, :df_coefs.shape[1]].tolist())
    df_inputs = df_inputs.iloc[:, :df_coefs.shape[1]]
    cols_inputs = df_inputs.columns.tolist()
    inputs = df_inputs.iloc[0, :].tolist()

    # print(df_inputs.head())
    bound = [float(i) * 3 for i in sigmas]
    lower_bound = [i - j for i, j in zip(inputs, bound)]
    upper_bound = [i + j for i, j in zip(inputs, bound)]

    ## Generate normally distributed inputs, set at 10% at the moment..
    ## TODO inputs are now based on an initial run, they should be based on the actual mu's and sigma's!!!
    # control individual bounds http://deap.readthedocs.io/en/master/tutorials/basic/part2.html#tool-decoration


    main()