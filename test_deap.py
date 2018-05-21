#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import array
import numpy as np

from deap import base
from deap import creator
from deap import tools
import pandas as pd
import array
import random
import matplotlib.pyplot as plt
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 10


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

elements = uniform(BOUND_LOW, BOUND_UP, NDIM)
print(elements)

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
print(toolbox.individual())

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
toolbox.register("select", tools.selNSGA2)


def main(seed=None):
    random.seed(seed)

    NGEN = 50
    MU = 20
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    print(pop)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done

    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)


    return pop, logbook


if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, logbook = main()

    fit_avgs = logbook.select("avg")

    df_fit = pd.DataFrame(fit_avgs)
    df_fit.plot()

    # pop.sort(key=lambda x: x.fitness.values)

    # print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    # import matplotlib.pyplot as plt
    # import numpy

    # front = numpy.array([ind.fitness.values for ind in pop])
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.axis("tight")
    # plt.show()
    plt.show()

"""

individual_mus = [8, 10, 0.4, 1.2]
individual_sigmas = [2, 3, 0.09, 0.2]

intercept = [123, 192, 123] # per objective
coefs = [[0.4, 0.6, 0.23, 0.16], [0.1, 0.9, 0.53, 0.26], [0.1, 0.9, 0.53, 0.26]] # per variable
targets = [300, 200, 200] # per objective

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=None,
               smin=None, smax=None, best=None)

print(len(individual_mus))
def fun(individual_mus, individual_sigmas):
    elements = []
    for x in range(len(individual_mus)):
        elements.append(random.gauss(individual_mus[x], individual_sigmas[x]))
    return elements

elements = fun(individual_mus, individual_sigmas)
print(elements)

def initParticle(pcls, size, pmin, pmax, smin, smax):
    part = pcls(random.gauss(pmin, pmax) for _ in individual_mus)
    part.speed = [random.gauss(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


toolbox = base.Toolbox()
toolbox.register("particle", fun, individual_mus, individual_sigmas)

print(toolbox.particle())



def objectives(individual):
    x, y = individual[0], individual[1]
    obj1 = 1-np.exp(-((x-1/np.sqrt(1))**2+(y-1/np.sqrt(2))**2))
    obj2 = 1-np.exp(-((x+1/np.sqrt(1))**2+(y+1/np.sqrt(2))**2))
    return obj1, obj2,

# problem dimension
NDIM = 2

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -4, 4)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", objectives)


def main():
    # differential evolution parameters
    CR = 0.25
    F = 1
    MU = 300
    NGEN = 200

    pop = toolbox.population(n=MU)

    pareto_front = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(fitnesses)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    #print(logbook.stream)

    for g in range(1, NGEN):
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F*(b[i]-c[i])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        pareto_front.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        print(logbook.stream)

    print("Best individual is ",
          pareto_front[0], pareto_front[0].fitness.values[0])

    return(pareto_front)

if __name__ == "__main__":
    pf = main()
    #print(pf)


# Create the item dictionary: item name is an integer, and value is
# a (weight, value) 2-uple.
NBR_ITEMS = 100
IND_INIT_SIZE = 5
items = {}
# Create random items and store them in the items' dictionary.
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.uniform(0, 100))

# Initialize a population and the individuals therein.
toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][0]
        value += items[item][1]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 10000, 0             # Ensure overweighted bags are dominated
    return weight, value

def cxSet(ind1, ind2):
    
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

def mutSet(individual):
    
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,


toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    NGEN = 50
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    print(stats)
    return pop, stats, hof

"""