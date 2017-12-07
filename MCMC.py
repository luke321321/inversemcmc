# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:33:52 2017

@author: s1002685

Will later optimise using Numba/Cython
"""
import numpy as np
import math, random
#import matplotlib.pyplot as plt
#import time

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

root2Pi = math.sqrt(2*math.pi)
normalDist = lambda x: math.exp(-x**2/2)/root2Pi

#Create a dummy kernal, just l^2 norm
kernal = lambda x,y: np.linalg.norm(x-y)

def createGaussianEmulator(phi,N,kernal,designPoints):
    #Kernal has to be able to give out same shape as it gets in
    phiStar = phi(designPoints)
    #Create K_*
    kernalStar = kernal(np.meshgrid(phiStar,phiStar)) #NxN matrix
    kernalStarInverse = np.linalg.inv(kernalStar)
    k = lambda u: kernal(u,phiStar) #N vector
    mean = k @ kernalStarInverse @ phiStar
    #kernalN = lambda x,y: kernal(x,y) - k(y) @ kernalStarInverse @ k(x)
    
    #At the moment just doing the simple case for \Phi_N(u) = mean(u)
    return mean     
    

#t0 = time.clock()
#x = MHRandomWalk(normalDist, length=1000000, speed=0.5)
#t1 = time.clock()
#print('CPU time for loops in Python:', t1-t0)
#
#plt.hist(x, normed=True, bins=77)
#plt.show()