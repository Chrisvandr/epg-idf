import operator
import random
import getpass
import time
import matplotlib.pyplot as plt
#import networkx as nx
import pandas as pd
#from scipy.optimize import minimize, basinhopping, least_squares, leastsq
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import matplotlib.pyplot as plt

def fun1():
    targets = [10, 12, 3]
    q = [11,8,12,12,9,10,12,7,12,3,22]
    k= [12,11,12,12,20,10,12,0,12,10,11]
    x2=np.arange(1,12,1)
    for i in range(4):
        scolors = ['red' if targets[0]-q[i]/10 > valuex < targets[0]+q[i]/10 and targets[1]-k[i]/10 > valuey < targets[1]+k[i]/10 else 'green' for valuex, valuey in zip(q,k)]
        print(scolors)

    scolors = [colors[0] if targets[i] - df_predicted[cols_objectives[i]] / 10 < valuex < targets[i] + df_predicted[cols_objectives[i]] / 10 and targets[j] - df_predicted[cols_objectives[j]] / 10 < valuey < targets[j] + df_predicted[cols_objectives[j]] / 10 else
               colors[1] for valuex, valuey in zip(df_predicted[cols_objectives[i]], df_predicted[cols_objectives[j]])]
    print(scolors)
    # Calculate an outlier limit (I chose 2 Standard deviations from the mean)
    k_bar = np.mean(k)
    outlier_limit = 2*np.std(k)
    # Generate a colour vector
    kcolors = ['red' if abs(value - k_bar) > outlier_limit else 'yellow' for value in k]

    #Plot using the colour vector
    plt.scatter(x2,k, label="signal", c = kcolors)
    print(kcolors)
    plt.show()




timeline0 = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + (
"0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(1, 1470, 1)]

print(timeline0)


timeline = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + ":" + (
"0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(30, 1470, 30)]
timeline =  timeline # one for weekday and weekendday
print(timeline)

timeline2 = [(str(i / 60) if i / 60 > 9 else "0" + str(i / 60)) + ":" + (str(i % 60) if i % 60 > 9 else "0" + str(i % 60)) for i in range(0, 1440, 30)]
print(timeline2)

timerange = range(0, 1440, 30)
print(timerange)

timerange2 = ['%s:%s' % (h, m) for h in ([00] + list(range(1, 24))) for m in ('00', '30')]
print(timerange2)


UserName = getpass.getuser()
DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/SURROGATE/'
DataPath_model_real = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/LegionRuns/'

#models = [("LR", lr), ("rr", rr)]
df_coefs = pd.read_csv(DataPath_model_real+ 'LR_coef.csv', header=0, index_col=0)
df_intercepts = pd.read_csv(DataPath_model_real+ 'LR_intercept.csv', header=0, index_col=0)
df_inputs = pd.read_csv(DataPath_model_real+ 'input_outputs.csv', header=0)

sigmas = df_inputs.iloc[0, :df_coefs.shape[1]]*(1/10)
sigmas = sigmas.tolist()
print(len(sigmas))
print(sigmas)

#"{0:0=2d}".format(a)

# for i in range(30, 1440, 30):
#     #print(i/60)
#     if i/60 < 10:
#         print("0" + str(int(i / 60)))
#         #print(str(int(i/60)))
#     else:
#         print(str(int(i / 60)))

