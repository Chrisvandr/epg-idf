import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping, least_squares, leastsq
import math
from numpy import arange, sin, pi, random, array
import operator
import random
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from pygmo import *

## DEAP ##


# http://bluebrain.github.io/eFEL/deap_optimisation.html



#http://stackoverflow.com/questions/22403982/how-to-minimize-a-function-using-deap
# https://gist.github.com/cmd-ntrf/7848947#file-multi-objectives_optimized_graph-py-L12
# https://08335959247544978238.googlegroups.com/attach/cf1e0d17adb3e/mb_differential_evolution_multi_objective_deap.py?part=0.1&view=1&vt=ANaJVrEYs7Vt8vaUD1QezRJ45cO3EBiv05N3O5WpunAgCaFGGYVARphlyuUjdloSnltlp0p4BI_3G06yok1B_x0gj6lWgS4MvqJy3xl1tjEYxKqqWUvZF-A

import array
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import networkx as nx
import math
#import drawG




'''

measured = {
    1: [0, 0.02735, 0.47265],
    6: [0.0041, 0.09335, 0.40255],
    10: [0.0133, 0.14555, 0.34115],
    20: [0.0361, 0.205, 0.2589],
    30: [0.06345, 0.23425, 0.20225],
    60: [0.132, 0.25395, 0.114],
    90: [0.2046, 0.23445, 0.06095],
    120: [0.2429, 0.20815, 0.04895],
    180: [0.31755, 0.1618, 0.02065],
    240: [0.3648, 0.121, 0.0142],
    315: [0.3992, 0.0989, 0.00195]
}


def A(x, a, k):
    return a * math.exp(-x * k)


def B(x, a, k, l):
    return k * a / (l - k) * (math.exp(-k * x) - math.exp(-l * x))


def C(x, a, k, l):
    return a * (1 - l / (l - k) * math.exp(-x * k) + k / (l - k) * math.exp(-x * l))


def f(x0):
    a, k, l = x0
    error = []

    for x in measured:
        #print(x)
        #print(measured[x][0])
        error += [C(x, a, k, l) - measured[x][0],
                  B(x, a, k, l) - measured[x][1],
                  A(x, a, k) - measured[x][2]
                  ]
        print(C(x,a,k,l)-measured[x][0])
    print(error)
    return error


def main():
    x0 = (0.46, 0.01, 0.001)  # initial parameters for a, k and l
    x, cov, infodict, mesg, ier = leastsq(f, x0, full_output=True, epsfcn=1.0e-2)
    print(x)


if __name__ == "__main__":
    main()

'''

#http://root42.blogspot.co.uk/2013/07/how-to-use-scipy-least-squares-to.html


# # an example function with multiple minima
# def f(x): return x.dot(x) + np.sin(np.linalg.norm(x) * np.pi)
#
# # the starting point
# x0 = [10., 10.]
#
# # the bounds
# xmin = [1., 1.]
# xmax = [11., 11.]
#
# # rewrite the bounds in the way required by L-BFGS-B
# bounds = [(low, high) for low, high in zip(xmin, xmax)]
#
# # use method L-BFGS-B because the problem is smooth and bounded
# minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
# res = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs)
#print(res)