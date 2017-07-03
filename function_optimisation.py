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
    model = joblib.load(DataPath_model_real + 'rr_model.pkl')
    individual = np.array(individual).reshape((1,-1))
    prediction = model.predict(individual)

    #prediction = prediction/(FLOOR_AREA)
    #print(prediction[0])

    #todo for diff buildings, the targets are different. The measured targets have to be comparable like for like with the predictions

    if BUILDING_ABR in {'CH'}:
        prediction = prediction[0]
        print(prediction)
        if TIME_STEP == 'year':
            prediction = [prediction[0]+sum(prediction[4:5])+prediction[7], sum(prediction[1:3]), prediction[6]] # Systems, L&P, Gas
            #print(prediction)
            # Fans, 56882.57295
            # Lights, 160068.5712
            # Equipment, 320129.8366
            # WaterSystems, 49546.05328
            # Cooling, 126392.1277
            # Heating, 106997.2152
            # Gas, 35210.47095
            # Pumps, 1320.103632
        elif TIME_STEP == 'month':
            prediction = prediction

    #print(prediction)
    return prediction

def evaluate(individual):
    diff = []
    prediction = calculate(individual)
    #print(prediction[0])
    #print(targets)
    for y in range(len(targets)):
        output = math.sqrt((targets[y] - prediction[y]) ** 2)
        #output = abs(targets[y] - prediction[0][y])
        diff.append(output)

    #normalize output
    #s = sum(diff)
    #dif f = [float(i)/s for i in diff]
    return diff

# PLOT: Plot sum off RMSE objective differences
def plot_sum_RMSE():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    gen = logbook.select("gen")
    for i, v in enumerate(best_inds_fitness):
        ax.plot(gen[i], sum(v), 'o', color='black', label='avg' if i == 0 else "")
        #ax3.plot(gen[i], sum(best_inds_fitness[i]), 'o', color='blue', label='max' if i == 0 else "")
        #ax3.plot(gen[i], sum(best_inds_fitness[i]), 'o', color='blue', label='min' if i == 0 else "")
    ax.legend(loc='best')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Generations')
    ax.set_title('ax')


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
def plot_avg_RMSE():
    df_avg = pd.DataFrame(fit_avgs, columns=cols_objectives)
    df_inds_fit = pd.DataFrame(best_inds_fitness, columns=cols_objectives)

    ax = df_avg.plot(color=colors, title='avg')
    ax.set_ylabel("RMSE or absolute difference")

def main():
    random.seed(20)
    #todo set at NGEN=444, popsize=44
    NGEN = 44 # is the number of generation for which the evolution runs
    # For selTournamentDCD, NGEN has to be a multiple of four the population size
    population_size = 44 # no of individuals/samples
    CXPB = 1 # is the probability with which two individuals are crossed

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)
    stats.register('std', np.std, axis=0)
    stats.register('avg', np.average, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "inputs" #, "max", "avg" #'inputs', 'std', 'avg', 'evals'

    # Create an initial population of size n.
    pop = toolbox.population(n=population_size)
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

    best_inds, best_inds_fitness = [], []
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        #print(pop), print(len(pop[0]))
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

        best_ind = tools.selBest(pop, 1)[0]
        #print('calculated prediction best individual', calculate(best_ind))
        best_inds.append(best_ind) # add the best individual for each generation
        best_inds_fitness.append(best_ind.fitness.values)
        logbook.record(gen=gen, inputs=best_ind.fitness.values,  evals=len(invalid_ind), **record) #, inputs=best_inds_fitness

        if gen % 20 == 0:
            print(gen, calculate(best_ind))
        #with open("logbook.pkl", "wb") as lb_file: pickle.dump(logbook, lb_file)

    print("  Evaluated %i individuals" % len(invalid_ind))
    print("-- End of (successful) evolution --")
    #best_ind = max(pop, key=lambda ind: ind.fitness)
    best_ind = tools.selBest(pop, 1)[0]

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    #print("Best individual is %s, %s" % (best_ind2, best_ind2.fitness.values))
    return pop, logbook, pareto_front, best_inds, best_inds_fitness

if __name__ == '__main__':
    UserName = getpass.getuser()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']

    BUILDING_ABR = 'CH'
    TIME_STEP = 'month' #'year', 'month'

    if BUILDING_ABR in {'CH'}:
        DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/Run1000/'
        FLOOR_AREA = 5876

        if TIME_STEP == 'year':
            targets = [284545, 265146, 44914]
        elif TIME_STEP == 'month':
            # 'Systems', 'L&P', 'Gas' x 8 months
            # However, the surrogate model predicts for 12 months, 8 objectives.
            targets = [25513.324062499945, 27644.416375000023, 3992.4, 26905.829875000007, 31679.298406249967, 4119.0, 30610.03975000001, 36493.14106249997, 5532.25, 39521.320750000006, 32003.899749999968, 9911.8, 40758.72162499995, 35495.32093749996, 7531.6, 36286.747687500014, 33060.69987499995, 5212.571428571428, 42278.15051562502, 36573.40829687499, 5103.2, 42671.43143750001, 32195.969687499986, 3512.0]
            print('no. of objectives', len(targets))
    DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'

    df_coefs = pd.read_csv(DataPath_model_real + 'rr_coef_'+TIME_STEP+'.csv', header=0, index_col=0) #todo use input csv instead?
    df_inputs = pd.read_csv(DataPath_model_real + 'input_outputs_'+TIME_STEP+'.csv', header=0)

    # TODO allow for excluding variables with little influence, model order reduction (but might not be necessary as long as optimisation works?)

    print(len(targets), df_coefs.shape[1])  # no outputs, no of variables
    cols = df_inputs.columns.tolist()
    cols_objectives = cols[df_coefs.shape[1]:df_coefs.shape[1] + len(targets)]
    print(cols_objectives)
    cols_objectives = ['Systems', 'L&P', 'Gas']

    sigmas = df_inputs.iloc[0, :df_coefs.shape[1]] * (1 / 10) #todo atm sigma set at 10% of inputs
    sigmas = sigmas.tolist()
    print('inputs', df_inputs.iloc[0, :df_coefs.shape[1]].tolist())
    inputs = df_inputs.iloc[0, :df_coefs.shape[1]].tolist()
    bound = [float(i) * 3 for i in sigmas]
    lower_bound = [i - j for i, j in zip(inputs, bound)]
    upper_bound = [i + j for i, j in zip(inputs, bound)]
    print('lower bound', lower_bound)

    # print(np.normalize(targets))
    print('weights normalized to target', [-float(i) / sum(targets) for i in targets])
    print('equal weights', [-1 for i in targets])
    creator.create('Fitness', base.Fitness, weights=[-float(i) / sum(targets) for i in targets])

    creator.create('Individual', array.array, typecode="d", fitness=creator.Fitness)  # set or list??

    toolbox = base.Toolbox()

    ## using custom function for input generation
    toolbox.register('expr', elements, df_inputs, lower_bound, upper_bound)
    ## importing the array type individual here, needed for evaluation
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)


    # TODO there needs to be constraint handling on the input parameters, because it's more realistic if some inputs don't change at all. Or for example the office heating setpoint seems to change to 31, which makes no sense in reality...
    # Constraint handling _> http://deap.gel.ulaval.ca/doc/dev/tutorials/advanced/constraints.html

    # Register functions to toolbox and use within main()
    # TODO try different selection criteria/mutate/mate

    toolbox.register('mate', tools.cxTwoPoint)
    # toolbox.register("mate", tools.cxUniform, indpb=0.1)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_bound, up=upper_bound, eta=20.0)

    # toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bound, up=upper_bound, eta=20.0, indpb=0.05) #TODO can't divide by zero, need to remove variables that are 0
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigmas,indpb=0.01)  # sigmas are set to a sequence, #TODO retrieve the right values, they are now based on 1/10 of some initial sample

    toolbox.register('select', tools.selNSGA2)
    # toolbox.register('select', tools.selSPEA2)
    toolbox.register('evaluate', evaluate)  # add the evaluation function



    pop, logbook, pareto_front, best_inds, best_inds_fitness = main()
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

    front = np.array([ind.fitness.values for ind in pop])
    #print(front)

    #print(df_inds_fit)
    print(best_inds[-1])
    print('targets', cols_objectives, targets)
    print('best individual prediction', calculate(best_inds[-1]))


    #print(pareto_front)
    #print(logbook)
    #print(pop)


    #plot_prediction_gen()
    #plot_sum_RMSE()
    plot_energy_objective() #shows each end use and how they change absolutely over the generations
    plot_avg_RMSE()

    plt.show()
