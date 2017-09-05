import numpy as np
import matplotlib.pyplot as plt
import math
import array
import random
import pickle
import pandas as pd
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.externals import joblib
from decimal import *


def elements(df_inputs, lower_bound, upper_bound):
    elements = [random.uniform(a, b) for a, b in zip(lower_bound, upper_bound)]
    #elements = []
    # for x in range(df_coefs.shape[1]):
    #         elements.append(random.gauss(df_inputs.iloc[0][x], df_inputs.iloc[0][x]*(1/10))) # set to 10% variation currently... #TODO change
    return elements

## Function for calculating the end uses based on the coefficients and intercepts
def calculate(individual):
    outputs = []
    for y in range(len(targets)):
        outputs.append(sum(np.array(coefficients[y])* individual)+intercepts[y])
        #outputs.append(output)
    outputs = np.array(outputs)

    return outputs

# Load the pickle surrogate model
# def calculate(individual):
#     model = joblib.load(DATAPATH_MODEL_REAL +  'rr_' + TIME_STEP + '_model.pkl')
#     individual = np.array(individual).reshape((1,-1))
#     prediction = model.predict(individual)[0]
#     prediction = prediction[:NO_OF_OBJECTIVES]
#
#     return prediction

def evaluate(individual):
    diff = []
    prediction = calculate(individual)

    for y in range(len(targets)):
        output = math.sqrt((targets[y] - prediction[y]) ** 2)
        #output = abs(targets[y] - prediction[y])
        diff.append(output)
    diff = diff[:NO_OF_OBJECTIVES]
    #diff = [float(i)/sum(diff) for i in diff] #normalize output

    return tuple(diff)

# PLOT: Plot energy use for the objectives with increasing generation
def plot_objectives():
    result = []
    for i, v in enumerate(best_inds):
        prediction = calculate(best_inds[i])
        result.append(prediction)

    df_result = pd.DataFrame(result, columns=cols_objectives)
    ax = df_result.plot(title='Prediction of the best individual per generation', color=colors)
    targets_dup = ([targets, ]) * len(result)  # duplicate targets list
    df_target = pd.DataFrame(targets_dup, columns=cols_objectives)
    df_target.plot(ax=ax, style='--', color=colors)

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

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    pop = toolbox.select(pop, POPULATION_SIZE) # no actual selection is done
    best_inds, best_inds_fitness = [], []
    record = stats.compile(pop)
    logbook.record(**record)  # inputs=best_inds_fitness
    hof.update(pop)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
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
        pop = toolbox.select(pop + offspring, len(offspring))

        best_ind = tools.selBest(pop, 1)[0]
        best_inds.append(best_ind) # add the best individual for each generation
        best_inds_fitness.append(best_ind.fitness.values)

        record = stats.compile(pop)
        logbook.record(gen=gen, inputs=[int(e) for e in best_ind.fitness.values], **record)
        hof.update(pop)
        if gen % 20 == 0:
            print(gen, 'best_ind_pred', [int(e) for e in calculate(best_inds[-1]).tolist()], 'targets', targets[:NO_OF_OBJECTIVES], 'fitness', [int(e) for e in best_ind.fitness.values], 'sum abs diff', [int(sum(targets)-sum(calculate(best_inds[-1])))], )

        # with open("logbook.pkl", "wb") as lb_file: pickle.dump(logbook, lb_file)

    return pop, hof, logbook, best_inds, best_inds_fitness

if __name__ == '__main__':
    NGEN = 500 # is the number of generation for which the evolution runs  # For selTournamentDCD, NGEN has to be a multiple of four the population size
    POPULATION_SIZE = 48 # no of individuals/samples
    CXPB = 0.9 # is the probability with which two individuals are crossed
    MUTPB = 0.6 # probability of mutating the offspring
    INDPB = 0.2 # Independent probability of each attribute to be mutated
    SIGMA_VARIATION = 20 #percentage of variation for the input variables
    BUILDING_ABR = 'CH'
    TIME_STEP = 'year' #'year', 'month'
    NO_OF_OBJECTIVES = 8 # number of objectives to optimise

    if BUILDING_ABR in {'CH'}:
        #DATAPATH_MODEL_REAL = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/Run1000/'
        END_USES = False

        inputs = [3.09, 4.15, 3.37, 3.24, 3.33, 2.57, 3.09, 4.15, 3.68, 5.12, 3.28, 3.05, 3.54, 3.89, 3.4, 5.63,
                  2.36, 2.97, 3.78, 3.71, 2.86, 2.3, 0.01, 7.29, 16.03, 12.72, 14.7, 10.84, 7.77, 8.16, 29.12,
                  9.65, 12.39, 12.13, 8.24, 15.76, 15.44, 16.94, 12.58, 4.81, 3.11, 8.32, 6.73, 6.73, 6.48, 6.4,
                  7.33, 23.19, 5.86, 9.58, 4.5, 5.33, 15.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, 0.82, 3.83, 12.3, 11.34, 3.02, 11.86,
                  6.62, 12.24, 18.12, 51.95, 15.0, 7.45, 54.68, 3.6, 0.0, 0.1, 0.16, 0.9, 0.91, 2.09, 1.98,
                  0.03, 0.04, 0.31, 0.15, 0.15, 0.18, 0.19, 0.14, 2.97, 192.55, 208.06, 22.5, 1.5, 1.38, 21.69,
                  29.76, 18.13, 3.0, 3.71, 0.0, 0.91, 0.0, 3.0, 17.25, 0.84, 1.73, 0.02, 3.0, 2.51, 1.05, 0.0,
                  0.13, 2.0, 11.51, 1.39, 0.83, 0.14, 2.0, 81.02, 41.09, 14.23, 4.05, 1.63, 1.11, 31.14, 29.26,
                  14.63, 14.63, 22.87, 14.84, 7.68, 7.42]

        if TIME_STEP == 'year':
            targets = [284545, 265146, 44914] # sum of 8 months of data for 3 energy end-uses, 'Systems', 'L&P', 'Gas' x 8 months
            cols_objectives = ['Systems', 'L&P', 'Gas']
            coefficients = [
                [388, -8915, -1537, -560, 996, -781, -801, -372, -2081, -978, -1202, -1579, -3003, -11389, -3506,
                 -1760, -1764, -4571, -1685, 464, -1639, 633, 43, 312, -26, 147, 16, 74, -310, 118, -99, 492, -143,
                 -188, -65, 95, 26, 170, -3, -1012, -1481, 73, -9, -84, -559, -483, -21, -74, 56, -176, -352, 359,
                 -125, -18, 29, 15, 22, 10, 75, 39, 157, 18, -4, 7, -17, -15, 4, 0, 0, 0, -1, 0, -1, -4349, -1773,
                 1140, -314, -9, 473, -309, -403, 433, 767, 47, -175, 155, 183, 923, 0, 950, 446, 6741, -2753, 8640,
                 703, 123, 44, 14, -168, 157, -119, -423, -875, 2665, -20, 38, -6, -14118, 2027, 958, 881, 985, 776,
                 2962, 0, -7586, 0, 978, -51, 2180, -2788, 640, -1200, 1004, 170, 0, -885, -3360, -129, -2578, 6189,
                 625, -5224, -729, -505, -2410, 5358, 3222, 3207, -6103, -133, -2486, -11, 5314, 1027, 7274, 6761],
                [509, 188, -162, -967, -613, -449, 529, -16, 313, -522, 886, 445, 53, -485, 195, -459, -648, -727,
                 1270, -304, -927, 205, -154, 1800, 229, 584, 332, 294, 393, 650, 447, 4480, -58, -252, 6, -31, -59,
                 178, -382, -689, -1864, -369, 117, 61, -8, -413, -365, 1, -360, -449, -264, -126, -145, -25, 21,
                 -4, -6, 19, -17, 19, -10, 22, -2, 7, 2, -24, 2, 0, 2, 0, 2, 2, 6, -3291, -397, 2920, 193, 724,
                 2830, 465, 733, 1067, 6406, 353, -263, 406, 452, 757, 0, 487, 241, 6729, -1053, 3050, -3256, 2,
                 -50, 1333, -302, -296, 1276, -1338, 1203, 2729, -46, 84, 0, -287, 401, 412, 297, -3, 108, 2009, 0,
                 -2220, 0, 1196, 914, -2418, -3044, 308, 748, 1930, 494, 0, 406, -761, 216, 4826, 2972, 519, 1267,
                 258, 178, 984, -676, -305, -189, 7131, 1204, 1255, 746, 1003, -1051, -178, -2327],
                [-161, -137, 1013, 292, 144, -123, -6, 69, -516, -298, 13, 68, 307, -205, -475, -175, 572, -155,
                 -851, 326, 832, -160, 83, -508, 51, -31, -341, 44, 75, -279, -59, -67, -35, 15, -9, 36, -72, -30,
                 21, -289, -92, 134, 223, -53, 82, -188, -157, -83, -44, 30, -32, 95, 15, 69, 16, 2, 51, -3, 11, 0,
                 21, 4, 32, 0, 0, 10, -10, 0, 0, 0, 0, 0, 0, -36177, 1210, -547, -83, -44, -300, 11, -47, 75, -114,
                 -4, -61, 229, 14, -52, 0, 36, 549, -683, -742, 223, 1644, 229, 60, -1013, -397, 629, -300, 324,
                 -322, 6145, -22, -23, 2, -975, -4, 136, -224, 14, -199, 21, 0, 22, 0, 166, 279, -1036, -1019, 264,
                 -133, 434, -1007, 0, -312, -1111, 39, -490, 939, 230, 775, -12, -8, -361, 1994, 919, 746, -641,
                 337, -88, 181, -778, -354, 33, 216]]
            intercepts = [363070, -274327, 93024]
        elif TIME_STEP == 'month':
            targets = [83785, 74560, 83954, 78379, 57150, 62704, 72635, 81437]
            targets = targets[:NO_OF_OBJECTIVES]
            cols_objectives = [1, 2, 3, 4, 9, 10, 11, 12]
            coefficients = [
                [29, -1934, 243, -419, 214, 46, -127, 30, -480, -334, -333, -321, 141, -3281, -380, -474, -144,
                 -835, -142, 30, -146, -182, 76, 194, 11, 75, -64, 14, -104, 11, 13, 352, -4, -2, -52, 8, -69, 53,
                 -34, -406, -382, -201, 111, -39, -141, -190, -5, -26, -39, 27, -5, 120, -7, 28, 6, -12, 21, -22,
                 68, 22, 140, 13, 25, -15, -25, -10, -15, 0, 0, 0, 0, 0, 2, -10976, -1379, 47, 34, -6, 251, 52,
                 -195, 53, 483, 23, -49, 122, 78, 368, 0, 25, -777, 1367, -154, 1752, 12, 299, 54, -1324, 312, 723,
                 -330, -1264, -128, 4357, -6, -5, -1, -2815, 517, 92, 89, 3, 487, 562, 0, -2779, 0, -392, 113,
                 -1020, 511, 778, 381, -264, 1231, 0, -208, 279, 390, -904, -459, 28, 307, -15, -10, 770, -800, -91,
                 251, 892, -1039, 1732, -434, -1097, 178, 1244, -14],
                [4, -1548, 25, -407, 142, 10, -130, -138, -442, -155, -195, -107, -130, -2073, -363, -273, -183,
                 -745, -112, 70, -25, -14, -35, 274, 31, 85, 18, 29, -26, 33, 53, 630, 23, 1, -5, -23, -28, 25, -7,
                 -235, -146, -23, 64, -143, -77, -70, -1, -2, 17, -45, -37, 70, 26, 29, 1, -3, 22, -8, 15, 13, 89,
                 15, 10, -4, -14, -4, -7, 0, 0, 0, 0, 0, 2, -7279, 135, 91, 14, 39, 72, -8, -36, 101, 838, 32, -10,
                 98, 84, 132, 0, 326, -607, 2045, -446, 1772, 420, 111, -103, 236, 354, 227, 564, -242, -330, 2437,
                 -4, 1, -3, -2167, 195, 25, 17, 106, 271, 431, 0, -1698, 0, 30, 120, -412, -311, 597, 127, 329, 82,
                 0, -134, 223, 195, 0, -407, -106, 243, -125, -86, 85, 1111, 918, 1160, 140, -38, -645, 47, -40,
                 968, 1478, 632],
                [-34, -1676, -175, -246, -42, -22, -297, -205, -438, -16, -420, -47, -334, -1516, -654, -179, -262,
                 -543, -209, 82, -176, 8, -107, 320, 32, 119, 49, 57, 40, 9, 92, 911, 11, -7, 3, -21, -2, 4, -5,
                 -169, -51, -45, 61, -85, -101, -31, 24, 14, -13, -65, -50, 23, 39, 17, 0, 0, 20, -1, -6, 8, 42, 13,
                 4, 4, -6, -5, 1, 0, 0, 0, 0, 0, 3, -4962, -47, 268, 14, 81, 132, -3, 68, 155, 1205, 41, 7, 97, 100,
                 67, 0, 266, -322, 1014, -369, 1666, 176, -26, -113, 705, 217, -244, 1048, -27, -382, 791, -1, 5,
                 -3, -1953, 125, -19, 8, 133, 215, 227, 0, -996, 0, 248, 144, 45, -738, 297, -35, 656, -592, 0, 0,
                 320, 236, 69, -698, -178, 31, -59, -40, 270, 150, 406, 694, 200, 305, -1291, 187, 1033, 884, 220,
                 908],
                [-52, -1563, -269, -131, -103, -15, -373, -142, -435, 3, -582, -3, -464, -1186, -749, -103, -337,
                 -340, -298, 85, -239, 19, -99, 288, 24, 107, 63, 56, 41, 2, 83, 832, 0, -8, 14, -13, 6, -2, -8,
                 -114, -7, -36, 54, -39, -83, -4, 25, 11, -19, -57, -43, -5, 49, 2, 0, 0, 10, 1, -6, 4, 6, 11, -1,
                 5, -1, -5, 5, 0, 0, 0, 0, 0, 2, -2901, -105, 317, 15, 80, 143, -4, 71, 151, 1144, 42, 14, 75, 96,
                 23, 0, 206, -180, 700, -366, 1475, 31, -88, -96, 860, 64, -308, 1055, 90, -264, 12, 2, 6, -2,
                 -1710, 54, -37, 7, 133, 146, 91, 0, -496, 0, 260, 134, 144, -759, 127, -104, 685, -721, 0, 155,
                 336, 184, 316, -761, -168, -75, -21, -14, 274, -248, 106, 323, 141, 424, -1108, 245, 1148, 728,
                 -101, 547],
                [-71, -230, -278, -54, -113, 1, -259, -90, -268, 14, -443, 21, -459, -230, -542, -8, -262, -87,
                 -219, 12, -132, 39, -105, 120, 10, 52, 34, 36, 18, -5, 42, 376, 9, -9, 17, -10, 2, -4, -1, -11, 39,
                 -10, 32, -24, -27, 8, 6, 5, -1, -36, -21, -21, 17, -4, 0, 2, 2, 5, -5, 0, -6, 4, -3, 2, 3, 1, 6, 0,
                 0, 0, 0, 0, 2, -523, 143, 172, -2, 32, 55, -7, 44, 80, 492, 44, 10, 37, 83, -12, 0, 133, -207, 436,
                 -37, 172, 16, -65, -58, 826, -93, -111, 478, 213, -40, -287, 0, 4, -2, -1546, -23, 44, -18, 132,
                 105, -13, 0, 1, 0, 103, 80, 159, -242, 24, -110, 504, -404, 0, 17, 172, 126, 377, -373, -107, -172,
                 5, 4, 54, -251, -30, 64, 35, 253, -768, 161, 899, 167, -220, 171],
                [-60, -358, -108, -156, -19, 29, -199, -126, -311, -5, -299, -1, -426, -910, -528, -68, -218, -368,
                 -157, 61, -73, 16, -81, 252, 35, 81, 39, 37, 29, 22, 71, 670, 10, -5, 14, -21, -3, 11, -11, -106,
                 -3, -19, 50, -61, -72, -21, 14, 2, 3, -36, -54, 0, 20, 6, 0, -1, 7, 0, -7, 1, 12, 9, 2, 1, -2, -2,
                 1, 0, 0, 0, 0, 0, 3, -1876, 58, 179, 13, 50, 99, -10, 76, 124, 876, 38, 1, 83, 86, 33, 0, 289,
                 -346, 971, -128, 650, 39, -27, -104, 456, 61, -13, 706, 14, -169, 450, -1, 4, -2, -1849, 52, 31,
                 -11, 103, 132, 165, 0, -633, 0, -22, 76, 81, -148, 252, -32, 467, -237, 0, 18, 258, 160, 388, -548,
                 -157, -129, -55, -38, 57, 388, 433, 617, 98, 245, -1087, 145, 920, 660, 357, 530],
                [-14, -758, 10, -302, 73, 11, -171, -227, -400, -60, -171, -92, -273, -1651, -509, -191, -205, -692,
                 -76, 76, -20, -18, -62, 285, 42, 95, 22, 16, 29, 32, 78, 744, 19, -1, 0, -25, -10, 17, -8, -201,
                 -82, -32, 64, -120, -101, -60, 27, 11, 1, -48, -59, 50, 24, 24, 0, 0, 20, -4, -6, 9, 64, 13, 9, -1,
                 -8, -5, -5, 0, 0, 0, 0, 0, 3, -5145, 14, 139, 9, 41, 93, -32, 71, 121, 956, 32, -4, 113, 84, 92, 0,
                 331, -528, 1558, -318, 1247, 141, 75, -107, 340, 280, -48, 782, -162, -468, 1623, -3, 4, -2, -1954,
                 150, 26, 9, 58, 241, 359, 0, -1379, 0, 13, 108, -127, -302, 478, 84, 374, -38, 0, -231, 305, 236,
                 8, -651, -189, 74, -107, -74, 38, 1021, 908, 1198, 43, 115, -1174, 81, 542, 1076, 903, 1042],
                [22, -1395, 93, -360, 173, 14, -84, -174, -319, -111, -44, -102, -44, -2202, -331, -225, -139, -786,
                 -20, 114, -5, -35, 43, 228, 32, 61, 0, 17, -19, 46, 29, 462, 15, 1, -6, -26, -27, 30, -19, -207,
                 -165, -45, 35, -133, -59, -105, 3, -7, 19, -31, -47, 67, 21, 24, 2, -2, 16, -11, 7, 7, 76, 10, 7,
                 -5, -13, -5, -8, 0, 0, 0, 0, 0, 2, -6950, -57, 17, 2, 11, 87, -35, -13, 71, 646, 34, -21, 87, 83,
                 114, 0, 277, -611, 2163, -127, 1961, 371, 150, -93, -174, 380, 198, 352, -413, -473, 2711, -3, 1,
                 -2, -1819, 238, 37, 13, 58, 162, 388, 0, -1344, 0, 18, 108, -370, -281, 578, 154, 195, 269, 0,
                 -231, 236, 218, -6, -422, -106, 498, -137, -95, 166, 1198, 922, 1122, -104, 4, -426, 5, -530, 1311,
                 1366, 755]]

            intercepts = [83257, 43057, 14637, 6245, 2808, 1978, 24104, 50176]

    sigmas = [i* (SIGMA_VARIATION / 100) for i in inputs]
    bound = [float(i) * 3 for i in sigmas] # min/max is 3*sigma
    lower_bound = [i - j for i, j in zip(inputs, bound)]
    upper_bound = [i + j for i, j in zip(inputs, bound)]

    print('weights normalized to target', tuple([-float(i) / sum(targets) for i in targets]))
    #print('equal weights', tuple([-1 for i in targets]))

    creator.create('Fitness', base.Fitness, weights=tuple([-float(i) / sum(targets) for i in targets]))
    creator.create('Individual', array.array, typecode='d', fitness=creator.Fitness)  # set or list??

    toolbox = base.Toolbox()
    ## using custom function for input generation
    toolbox.register('expr', elements, inputs, lower_bound, upper_bound)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # TODO try different selection criteria/mutate/mate
    toolbox.register('mate', tools.cxTwoPoint)
    #toolbox.register("mate", tools.cxUniform, indpb=INDPB)
    #toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lower_bound, up=upper_bound, eta=20.0)

    # toolbox.register('mutate', tools.mutFlipBit, indpb=INDPB)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bound, up=upper_bound, eta=20.0, indpb=INDPB)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigmas, indpb=INDPB)  # sigmas are set to a sequence

    toolbox.register('select', tools.selNSGA2)
    #toolbox.register('select', tools.selSPEA2)

    toolbox.register('evaluate', evaluate)  # add the evaluation function
    #toolbox.register("evaluate", benchmarks.zdt1)

    pop, hof, logbook, best_inds, best_inds_fitness = main()

    logbook.chapters["fitness"].header = "min", "max", "avg"
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")


    print('HOF', len(hof))
    print('cols_targets', cols_objectives)
    print('best individual', [int(e) for e in calculate(best_inds[-1]).tolist()])
    print('targets', targets)
    print('absolute diff', [i - j for i, j in zip([int(e) for e in calculate(best_inds[-1]).tolist()], targets)])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a',
              '#c49c94', '#ff9896', '#c5b0d5', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd',
              '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5']
    plot_objectives() #shows each end use and how they change absolutely over the generations

    plt.show()
