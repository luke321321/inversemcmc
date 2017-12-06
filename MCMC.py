# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:33:52 2017

@author: s1002685
"""
import numpy as np
import math, random
import matplotlib.pyplot as plt
import time

def MHRandomWalk(distribution,length,speed=0.5,x0=0,burnTime=200):
    x = np.zeros(burnTime + length)
    x[0] = x0
    distOld = distribution(x0)
    for i in range(1,burnTime + length):
        y = x[i-1] + random.normalvariate(0,speed)
        if(distOld == 0): #Accept whatever
            x[i] = y
            distOld = distribution(y)
        else:
            u = random.random()
            distNew = distribution(y)
            if(u <= min(1,distNew/distOld)): #check acceptance
                x[i] = y
                distOld = distNew
            else:
                x[i] = x[i-1]
    return x

def normalDist(x):
    return math.exp(-x**2/2)/math.sqrt(2*math.pi)

t0 = time.clock()
x = MHRandomWalk(normalDist, length=100000, speed=0.5)
t1 = time.clock()
print('CPU time for loops in Python OLD METHOD:', t1-t0)

plt.hist(x, normed=True, bins=77)
plt.show()