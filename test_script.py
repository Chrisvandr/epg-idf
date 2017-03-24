__author__ = 'cvdronke'

from pyDOE import *

design = lhs(4, samples=10)
from scipy.stats.discributions import norm
means = [1,2,3,4]
stdvs = [.1,.5,1,.25]
for i in xrange(4):
    design[:,i]=norm(loc=means[i], scale=stdvs[i]).ppf(design[:,i])

print design