# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:33:52 2017

@author: s1002685
"""
import numpy as np
import math, random
import matplotlib.pyplot as plt

def MHRandomWalk(distribution,length,speed=0.5,x0=0,burnTime=200):
    x = np.zeros(burnTime + length)
    x[0] = x0
    for i in range(1,burnTime + length):
        y = x[i-1] + random.normalvariate(0,speed)
        u = random.random()
        if(u <= acceptance(distribution,x[i-1],y)):
            x[i] = y
        else:
            x[i] = x[i-1]
    return x
    
def acceptance(distribution,X,Y):
    if(distribution(X) == 0):
        return 1
    else:
        alpha = distribution(Y)/distribution(X)
        return min(1,alpha)

def normalDist(x):
    return math.exp(-x**2/2)/math.sqrt(2*math.pi)

x = MHRandomWalk(normalDist, length=10000, speed=0.5)
plt.hist(x, normed=True, bins=77)
plt.show()