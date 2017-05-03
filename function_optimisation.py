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
#import networkx as nx
import pandas as pd
#from scipy.optimize import minimize, basinhopping, least_squares, leastsq
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import algorithms

individual_mus = [8, 10, 0.4, 1.2]
individual_sigmas = [2, 3, 0.09, 0.2]
intercept = [2, 1, 3] # per objective
coefs = [[0.4, 0.6, 0.23, 0.16], [0.1, 0.9, 0.53, 0.26], [0.6, 0.1, 0.25, 0.2]] # per variable
targets = [5, 2, 3] # per objective
targets = [0.728198335,	0.478837711,	1.507051668,	1.688731628,	0.21321589,	3.550106898,	1.180567061,	0.018584588]

UserName = getpass.getuser()
DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'
DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/LegionRuns/'

#models = [("LR", lr), ("rr", rr)]
df_coefs = pd.read_csv(DataPath_model_real+ 'LR_coef.csv', header=0, index_col=0)
df_intercepts = pd.read_csv(DataPath_model_real+ 'LR_intercept.csv', header=0, index_col=0)
df_inputs = pd.read_csv(DataPath_model_real+ 'input_outputs.csv', header=0)

#print(df_coefs.iloc[0])
print(df_coefs.shape[0], df_coefs.shape[1]) # no outputs, no of variables

## Function for calculating the energy end uses based on the coefficients and intercepts from machine learning (The surrogate model)
def calculate(individual):
    outputs = []
    for y in range(df_coefs.shape[0]):
        output = []  # refill every parameter iteration
        for x in range(df_coefs.shape[1]):  # for each input variable in the individual/parameter inputs
            output.append(df_coefs.iloc[y][x] * individual[x])
        output = sum(output)
        output = output + df_intercepts.iloc[y][0]
        outputs.append(output)  # diff
    return outputs

## Generate normally distributed inputs, set at 10% at the moment..
def elements(df_inputs):
    elements = []
    for x in range(df_coefs.shape[1]):
            elements.append(random.gauss(df_inputs.iloc[0][x], df_inputs.iloc[0][x]*(1/10))) # set to 10% variation currently...

    #print(elements[113:120])
    return elements

sigmas = df_inputs.iloc[0, :df_coefs.shape[1]]*(1/10)
sigmas = sigmas.tolist()

creator.create('Fitness', base.Fitness, weights=(-1.0,)*df_coefs.shape[0]) # number of objective functions
creator.create('Individual', array.array, typecode="d", fitness=creator.Fitness) #set or list??

toolbox = base.Toolbox()
## random number generation
#toolbox.register('attr_float', random.randrange, len(individual_mus))
#toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(individual_mus))

## using custom function for input generation
toolbox.register('expr', elements, df_inputs)
## importing the array type individual here, needed for evaluation
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

#print(toolbox.individual()[1], coefs[1][1])

#TODO there is something wrong with the cleaner hot water schedule coefficients... I could remove them, they should have minor influence on energy

# evaluate based on the coefficients and intercept
# Constraint handling _> http://deap.gel.ulaval.ca/doc/dev/tutorials/advanced/constraints.html
def evaluate(individual):
    ## The machine learning created function (each output and variable has a separate coefficient)
    ##          outp1_c outp2_c outp[i]_c   variable    outp[i]_result
    ## coef1    x1o1_c  x1o2_c  x1_o[i]_c   x1          x1*x1o[i]_c
    ## coef2    x2o1_c  x2o2_c  x2_o[i]_c   x2          x2*x2o[i]_c
    ##                                                  sum + intercept outp[i]
    ## This is then evaluated against the actual data and minimized.
    diff = []
    outputs = []
    for y in range(df_coefs.shape[0]): # number of outputs
        output = []
        for x in range(df_coefs.shape[1]): # for each input variable in the individual/parameter inputs
            output.append(df_coefs.iloc[y][x] * individual[x]) # multiply the surrogate model coefficient by the input variable
        output = sum(output)
        output = output + df_intercepts.iloc[y][0]
        outputs.append(output)
        output = (targets[y] - output) ** 2 # take the square root and minimize to 0
        diff.append(output)
    return diff

#print(toolbox.individual())
#evaluate(toolbox.individual())

#Register functions to toolbox and use within main()
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigmas, indpb=0.2) # sigmas are set to a sequence, #TODO retrieve the right values, they are now based on 1/10 of some initial sample
toolbox.register('select', tools.selNSGA2)
toolbox.register('evaluate', evaluate) # add the evaluation function

def main():
    random.seed(20)

    NGEN = 40 # is the number of generation for which the evolution runs
    # For selTournamentDCD, NGEN has to be a multiple of four the population size
    population_size = 10 # no of individuals/samples
    CXPB = 0.9 # is the probability with which two individuals are crossed

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)
    stats.register('std', np.std, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max" #'inputs', 'std', 'avg', 'evals'


    # Create an initial population of size n.
    pop = toolbox.population(n=population_size)
    #print(pop)
    pareto_front = tools.ParetoFront()

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    #print(toolbox.individual()) # which values
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    pop = toolbox.select(pop, len(pop)) # no actual selection is done

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    best_inds = []
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and then mutate the offspring
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2) # crossover randomly chosen individual within the population

            toolbox.mutate(ind1) # mutates several attributes in the individual
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, population_size)
        record = stats.compile(pop)
        fits = [ind.fitness.values[0] for ind in pop]

        best_inds.append(tools.selBest(pop, 1)[0]) # add the best individual for each generation
        best_ind_stream = best_ind = max(pop, key=lambda ind: ind.fitness)
        logbook.record(gen=gen, evals=len(invalid_ind), **record) #, inputs=best_ind_stream
        print(logbook.stream)

        #with open("logbook.pkl", "wb") as lb_file: pickle.dump(logbook, lb_file)


    #print("Best individual is ", pareto_front[0], pareto_front[0].fitness.values[0])
    print("  Evaluated %i individuals" % len(invalid_ind))
    print("-- End of (successful) evolution --")
    best_ind = max(pop, key=lambda ind: ind.fitness)
    #best_ind2 = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    #print("Best individual is %s, %s" % (best_ind2, best_ind2.fitness.values))
    #print("Final population hypervolume is %f" % benchmarks.hypervolume(pop, [11.0, 11.0]))
    # hypervolume is not yet available, used in some scripts, but in the 1.1 dev version.
    return pop, logbook, pareto_front, best_inds

if __name__ == '__main__':
    pop, logbook, pareto_front, best_inds = main()
    logbook.chapters["fitness"].header = "min", "max"

    #TODO when tolerance is achieved (say 5% within result), i can calculate for example the CV(RMSE) by combining all results from a generation with the target objectives.
    #TODO http://deap.readthedocs.io/en/master/api/tools.html#deap.tools.ParetoFront
    #TODO plot best fitness in each generation. (which may sometimes be worse than the next gen)

    gen = logbook.select("gen")
    fit_maxs = logbook.select("max")
    print(gen)
    #print('best inds', best_inds)

    # plot objective differences...
    fig, ax1 = plt.subplots()
    result = []
    for i,v in enumerate(best_inds):
        prediction = calculate(best_inds[i])
        result.append(prediction)
        #print('best indices used array:', best_inds[i])
        #print('best indices used for prediction', prediction)

    for i,v in enumerate(result):
        alp = 1-(i/50)
        plt.plot(v, color='blue', label=gen[i], alpha=alp)

    plt.plot(targets, 'o', color='black', label='Target')
    plt.legend(loc='best')
    plt.show()

    #print(pareto_front)
    #print(logbook)
    #print(pop)
# control individual bounds http://deap.readthedocs.io/en/master/tutorials/basic/part2.html#tool-decoration