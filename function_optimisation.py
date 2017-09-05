import numpy as np
import matplotlib.pyplot as plt
import math
import array
import operator
import random
import getpass
import time
import pickle
import pandas as pd
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.externals import joblib
from decimal import *

# Generate normally distributed inputs, set at 10% at the moment..
## TODO inputs are now based on an initial run, they should be based on the actual mu's and sigma's!!!
# control individual bounds http://deap.readthedocs.io/en/master/tutorials/basic/part2.html#tool-decoration
def elements(df_inputs, lower_bound, upper_bound):
    elements = [random.uniform(a, b) for a, b in zip(lower_bound, upper_bound)]
    #elements = []
    # for x in range(df_coefs.shape[1]):
    #         elements.append(random.gauss(df_inputs.iloc[0][x], df_inputs.iloc[0][x]*(1/10))) # set to 10% variation currently... #TODO change
    return elements

# Load the pickle surrogate model made by 'surrogate_model.py' and calculate with new inputs
def calculate(individual):
    model = joblib.load(DATAPATH_MODEL_REAL +  'rr_' + TIME_STEP + '_model.pkl')
    individual = np.array(individual).reshape((1,-1))
    prediction = model.predict(individual)[0]

    #prediction = prediction/(FLOOR_AREA)
    #todo for diff buildings, the targets are different. The measured targets have to be comparable like for like with the predictions

    # if BUILDING_ABR in {'CH'}:
    #     if TIME_STEP == 'year':
    #         #print(prediction)
    #         #surrogate model calculates for 8 months 8 end-uses, so need to sum to compare with measurements.
    #         prediction = [prediction[0]+sum(prediction[4:6])+prediction[7], sum(prediction[1:4]), prediction[6]] # Systems, L&P, Gas
    #
    #     elif TIME_STEP == 'month':
    #         prediction = prediction

    prediction = prediction[:NO_OF_OBJECTIVES]
    #print(prediction)
    return prediction

def evaluate(individual):
    diff = []
    prediction = calculate(individual)

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
    diff = diff[:NO_OF_OBJECTIVES]
    #diff = [float(i)/sum(diff) for i in diff]
    #print(tuple(diff))
    return tuple(diff)

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
def plot_energy_objective():
    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i])
        result.append(prediction)

    df_result = pd.DataFrame(result, columns=cols_objectives)
    ax = df_result.plot(title='Prediction of the best individual per generation', color=colors)
    targets_dup = ([targets, ]) * len(result)  # duplicate targets list
    df_target = pd.DataFrame(targets_dup, columns=cols_objectives)
    df_target.plot(ax=ax, style='--', color=colors)

# PLOT: Plot objective differences using RMSE
def plot_best_fit():
    #df_avg = pd.DataFrame(fit_avgs, columns=cols_objectives)
    #df_mins = pd.DataFrame(fit_mins, columns=cols_objectives)
    df_inds_fit = pd.DataFrame(best_inds_fitness, columns=cols_objectives)

    ax = df_inds_fit.plot(color=colors)
    ax.set_title("Best Individual fitnesses per gen")

# PLOT: Plot objective differences using RMSE
def plot_fit():
    #df_avg = pd.DataFrame(fit_avgs, columns=cols_objectives)
    df_mins = pd.DataFrame(fit_mins, columns=cols_objectives)
    #df_inds_fit = pd.DataFrame(best_inds_fitness, columns=cols_objectives)

    ax = df_mins.plot(color=colors)
    ax.set_title("Minimal fitnesses over population per gen")


#https://github.com/DEAP/notebooks
#https://github.com/lmarti/evolutionary-computation-course/blob/master/AEC.06%20-%20Evolutionary%20Multi-Objective%20Optimization.ipynb
def main():
    random.seed(20)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)
    stats.register('std', np.std, axis=0)
    stats.register('avg', np.average, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "inputs" #, "max", "avg" #'inputs', 'std', 'avg', 'evals'

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
    pop = toolbox.select(pop, POPULATION_SIZE) # no actual selection is done
    best_inds, best_inds_fitness = [], []
    record = stats.compile(pop)
    logbook.record(**record)  # , inputs=best_inds_fitness
    hof.update(pop)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        #print(pop), print(len(pop[0]))
        #print([ind.fitness.valid for ind in pop if ind.fitness.valid])

        offspring = tools.selTournamentDCD(pop, len(pop)) # only works with "select" to NSGA-II
        #offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring] # Clone the selected individuals

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]): # Crossover and mutate offspring
            #print('inds', ind1, ind2)
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2) # crossover randomly chosen individual within the population

                # toolbox.mutate(ind1)
                # toolbox.mutate(ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for mutant in offspring:
            if random.random() <= MUTPB: # which offspring are mutated
                #mutate a subset of the attributes (variables) in the individuals, based on a percentage of the pop
                #for i in range(round(len(pop[0])*PERC_ATTR_TO_MUTATE)):
                toolbox.mutate(mutant) # mutates several attributes in the individual
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #print(len(invalid_ind))
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, len(offspring)) #todo, length of offspring or pop?

        #fits = [ind.fitness.values[0] for ind in pop]
        best_ind = tools.selBest(pop, 1)[0]
        best_inds.append(best_ind) # add the best individual for each generation
        best_inds_fitness.append(best_ind.fitness.values)

        record = stats.compile(pop)
        logbook.record(gen=gen, inputs=[int(e) for e in best_ind.fitness.values], **record)
        hof.update(pop)
        if gen % 20 == 0:

            print(gen, int(sum(targets)-sum(calculate(best_inds[-1]))), 'best_inds', [int(e) for e in calculate(best_inds[-1]).tolist()], 'targets', targets[:NO_OF_OBJECTIVES], 'fitness', [int(e) for e in best_ind.fitness.values])

        #with open("logbook.pkl", "wb") as lb_file: pickle.dump(logbook, lb_file)

    #best_ind = tools.selBest(pop, 1)[0] # best_ind = max(pop, key=lambda ind: ind.fitness)
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # print(logbook.stream)

    return pop, hof, logbook, best_inds, best_inds_fitness

if __name__ == '__main__':
    NGEN = 1000 # is the number of generation for which the evolution runs  # For selTournamentDCD, pop size has to be multiple of four
    POPULATION_SIZE = 24 # no of individuals/samples
    CXPB = 0.9 # is the probability with which two individuals are crossed
    MUTPB = 0.5 # probability of mutating the offspring
    INDPB = 0.8 # Independent probability of each attribute to be mutated
    #PERC_ATTR_TO_MUTATE = 0.3 #no of attributes to mutate per individual
    SIGMA_VARIATION = 20 #percentage of variation for the input variables
    BUILDING_ABR = 'CH' #'CH', 'MPEB'
    TIME_STEP = 'month' #'year', 'month'
    NO_OF_OBJECTIVES = 8

    if BUILDING_ABR in {'CH'}:
        DATAPATH_MODEL_REAL = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/Run1000/'
        FLOOR_AREA = 5876
        END_USES = False

        if TIME_STEP == 'year':
            targets = [284545, 265146, 44914] # sum of 8 months of data for 3 energy end-uses, 'Systems', 'L&P', 'Gas' x 8 months

        elif TIME_STEP == 'month':
            # todo WARNING, targets are starting from Sept-Dec then Jan-Apr (sept-dec are from 16, not 17, whereas sim files are all 16, so turn around manually for now)
            if END_USES == True:
                targets = [40758, 35495, 7531, 36286, 33060, 5212, 42278, 36573, 5103, 42671, 32195, 3512, 25513, 27644, 3992, 26905, 31679, 4119, 30610, 36493, 5532, 39521, 32003, 9911]
            elif END_USES == False:
                # [jan, feb, mar, apr, sep, oct, nov, dec]
                targets = [83785, 74560, 83954, 78379, 57150, 62704, 72635, 81437]

                # #todo why are the prediction when running the surrogate model so different from the targets??? something going wrong here i think or is this just the range it can predict in, it is from a single run.
                # singleprediction[92240.45006702337, 78637.6605726759, 81411.5389778027, 72367.14122711583, 40051.313988098904, 56626.15939919354, 73364.38556002386, 73735.20007448267]
                # singletestdata[90511.33516526977, 77031.2684885664, 79573.42786600048, 70828.36384547835, 39466.62852737905, 55644.990233075674, 72172.93111136527, 73036.75992940296]

                targets = targets[:NO_OF_OBJECTIVES]

            print('no. of objectives', len(targets))

    df_coefs = pd.read_csv(DATAPATH_MODEL_REAL + 'rr_coef_'+TIME_STEP+'.csv', header=0, index_col=0) #todo use input csv instead?
    df_inputs = pd.read_csv(DATAPATH_MODEL_REAL + 'input_outputs_'+TIME_STEP+'.csv', header=0)

    # TODO allow for excluding variables with little influence, model order reduction (but might not be necessary as long as optimisation works?)
    print(len(targets), df_coefs.shape[1])  # no outputs, no of variables
    cols = df_inputs.columns.tolist()
    cols_objectives = cols[df_coefs.shape[1]:df_coefs.shape[1] + len(targets)]
    print(cols_objectives)

    sigmas = df_inputs.iloc[0, :df_coefs.shape[1]] * (SIGMA_VARIATION / 100) #TODO ATM SIGMA SET AT 10% OF THE FIRST RUN OF INPUTS!!!!
    sigmas = sigmas.tolist()
    print('SIGMA', sigmas)
    print('inputs', df_inputs.iloc[0, :df_coefs.shape[1]].tolist())
    inputs = df_inputs.iloc[0, :df_coefs.shape[1]].tolist()
    bound = [float(i) * 3 for i in sigmas]
    lower_bound = [i - j for i, j in zip(inputs, bound)]
    upper_bound = [i + j for i, j in zip(inputs, bound)]
    print('lower bound', lower_bound)

    print('weights normalized to target', tuple([-float(i) / sum(targets) for i in targets]))
    print('equal weights', tuple([-1.0 for i in targets]))
    creator.create('Fitness', base.Fitness, weights=tuple([-1.0 for i in targets]))
    creator.create('Individual', array.array, typecode='d', fitness=creator.Fitness)  # set or list??

    toolbox = base.Toolbox()
    ## using custom function for input generation
    toolbox.register('expr', elements, df_inputs, lower_bound, upper_bound)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)


    # TODO there needs to be constraint handling on the input parameters, because it's more realistic if some inputs don't change at all. Or for example the office heating setpoint seems to change to 31, which makes no sense in reality...
    # Constraint handling _> http://deap.gel.ulaval.ca/doc/dev/tutorials/advanced/constraints.html

    # TODO try different selection criteria/mutate/mate
    toolbox.register('mate', tools.cxTwoPoint)
    #toolbox.register("mate", tools.cxUniform, indpb=INDPB)
    #toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_bound, up=upper_bound, eta=20.0)

    # toolbox.register('mutate', tools.mutFlipBit, indpb=INDPB)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bound, up=upper_bound, eta=20.0, indpb=INDPB) #TODO can't divide by zero, need to remove variables that are 0
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigmas, indpb=INDPB)  # sigmas are set to a sequence, #TODO retrieve the right values, they are now based on 1/10 of some initial sample

    toolbox.register('select', tools.selNSGA2)
    #toolbox.register('select', tools.selSPEA2)

    toolbox.register('evaluate', evaluate)  # add the evaluation function
    #toolbox.register("evaluate", benchmarks.zdt1)

    pop, hof, logbook, best_inds, best_inds_fitness = main()

    logbook.chapters["fitness"].header = "min", "max", "avg"

    #TODO when tolerance is achieved (say 5% within result), i can calculate for example the CV(RMSE) by combining all results from a generation with the target objectives.
    #TODO can I figure out how far the optimised result input values differ from the base input values?
    #TODO Because the predicted data for CH for example in its solution space does not include the measured data point for systems, but is still able to optimise.
    #TODO http://deap.readthedocs.io/en/master/api/tools.html#deap.tools.ParetoFront
    #TODO plot best fitness in each generation. (which may sometimes be worse than the next gen)

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    #print('best inds', best_inds)
    #sorted(individuals, key=attrgetter("fitness"), reverse=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    #plot:
    #http://deap.readthedocs.io/en/master/tutorials/basic/part3.html
    print('HOF', len(hof))
    front = np.array([ind.fitness.values for ind in pop])

    #print(front)

    print(best_inds[-1])
    print('cols_targets', cols_objectives)
    print('best individual prediction vs. targets')

    print('best individual', [int(e) for e in calculate(best_inds[-1]).tolist()])
    print('targets', targets)
    print('absolute diff', [i - j for i, j in zip([int(e) for e in calculate(best_inds[-1]).tolist()], targets)])


    #print(logbook)
    #print(pop)


    #plot_prediction_gen()
    plot_energy_objective() #shows each end use and how they change absolutely over the generations
    #plot_fit()
    plot_best_fit()

    plt.show()
