# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:33:52 2017

@author: s1002685

Will later optimise using Numba/Cython
"""
import numpy as np
import math, random
import matplotlib.pyplot as plt
import time

#We pass dimension via x0
def MHRandomWalk(density,length,speed=0.5,x0=0,burnTime=200):
    x = np.zeros((burnTime + length, x0.size))
    #Generates normal rv in R^n of length=speed or 0.
    rvNormal = speed*normalizeVector(np.random.randn(burnTime + length, x0.size))
    #TODO update so that it's multi-dimensional

    x[0] = x0
    densityOld = density(x0)
    for i in range(1,burnTime + length):
        y = x[i-1] + rvNormal[i]
        if(densityOld == 0): #Accept whatever
            x[i] = y
            densityOld = density(y)
        else:
            u = random.random()
            densityNew = density(y)
            if(u <= min(1,densityNew/densityOld)): #check acceptance
                x[i] = y
                densityOld = densityNew
            else:
                x[i] = x[i-1]
    return x[200:]


root2Pi = math.sqrt(2*math.pi)
normalDensity = lambda x: math.exp(-np.sum(x**2)/2)/root2Pi

def createGaussianEmulator(phi,kernal,designPoints):
    #Kernal has to be able to give out same shape as it gets in
    phiStar = phi(designPoints)
    #Create K_*
    kernalStar = kernal(np.meshgrid(designPoints,designPoints)[0],
                        np.meshgrid(designPoints,designPoints)[1]) #NxN matrix
    kernalStarInverse = np.linalg.inv(kernalStar)
#    k = lambda u: kernal(u,phiStar) #N vector
    kernalStarInverseDotPhiStar = kernalStarInverse @ phiStar
    #might need to have this vectorised for more dimensions
    mean = lambda u: kernal(u,designPoints) @ kernalStarInverseDotPhiStar
    #kernalN = lambda x,y: kernal(x,y) - k(y) @ kernalStarInverse @ k(x)
    
    #At the moment just doing the simple case for \Phi_N(u) = mean(u)
    return mean     

def normalizeVector(a):
    """Normalises the vector a but keeps 0 vectors as 0 vectors"""
    norm = np.sum(a*a, 1)
    for x in np.nditer(norm,op_flags=['readwrite']):
        if(x == 0):
            x[...] = 1
    return a/(np.sqrt(norm)[:,None])

#%%Test case with easy phi.  G = identity
#First neeed to generate some data y
#setup
np.random.seed(1564)
sigma = 1
dimensionU = 1
length = 10**5 #length of random walk in MCMC
numberDesignPoints = 100
speedRandomWalk = 0.5

#Generate data
u = np.random.randn(dimensionU)
y = u + sigma*np.random.standard_normal(u.shape)

phi = np.vectorize(lambda u: np.linalg.norm(y-u)/(2*sigma))
#Create a dummy kernal, just l^2 norm
kernal = np.vectorize(lambda x,y: np.linalg.norm(x-y))

#N design points in [-2,2]
designPoints = np.linspace(-2,2,numberDesignPoints)
GP = createGaussianEmulator(phi,kernal,designPoints)

#Plotting phi and GP of phi:
#t = np.arange(-2,2,0.1)
#plt.plot(t,GP(t))
#plt.plot(t,phi(t), color='green')
#plt.plot(designPoints,phi(designPoints), 'ro')
#plt.show()

#%% Calculations
densityPrior = lambda u: normalDensity(u)*np.exp(-phi(u))
densityPost = lambda u: normalDensity(u)*np.exp(-GP(u))

#print('Running MCMC with length:', length)
#t0 = time.clock()
#distPrior = MHRandomWalk(densityPrior, length, speed=speedRandomWalk)
#t1 = time.clock()
#print('CPU time calculating distPrior:', t1-t0)
#
#t0 = time.clock()
#distPost = MHRandomWalk(densityPost, length, speed=speedRandomWalk)
#t1 = time.clock()
#print('CPU time calculating distPost:', t1-t0)




#%% Plotting distributions of Prior and Post
#plt.hist(distPrior, bins=77, alpha=0.5, density=True, label='Prior')
#plt.hist(distPost, bins=77, alpha=0.5, density=True, label='Post')
#plt.legend(loc='upper right')
#plt.show()







#%% Tests

#%% Testing Metroplis-Hastings algorithm
#t0 = time.clock()
#x = MHRandomWalk(normalDensity, length=100000, speed=0.5, x0=np.array([0,0,0,0,0]))
#t1 = time.clock()
#print('CPU time for loops in Python', t1-t0)
#plt.hist(x, bins=77,  density=True)
#plt.show()

#%% Check normaliseVector
#print('Checking normaliseVector code')
#a = np.array([[0,1],[0,0],[2,3]])
#b = normalizeVector(a)
#ans = np.array([[0,1],[0,0],[2/math.sqrt(13), 3/math.sqrt(13)]])
#print('normaliseVector, Test 1:', np.allclose(b,ans))