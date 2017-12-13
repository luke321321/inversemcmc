# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:33:52 2017

@author: s1002685

Will later optimise using Numba/Cython
"""
import numpy as np
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import time
#import timeit
#import cProfile, pstats

#We pass dimension via x0
def MHRandomWalk(density, length, speed=0.5, x0=np.array([0]), burnTime=200):
    if isinstance(x0, np.ndarray):
        dim = x0.size
    else:
        dim = 1
        
    x = np.zeros((burnTime + length, dim))
    #Generates normal rv in R^n of length=speed or 0.
    rvNormal = speed*normalizeVector(np.random.randn(burnTime + length, dim))

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
normalDensity = lambda x: math.exp(-np.dot(x,x)/2)/root2Pi

def GaussianEmulator_Gauss(phi, designPoints):
    """Creates a Guassian Emulator with the kernel being a Guassian.
    Currently just sets the Guassian Emulator to be the mean"""
    phiStar = phi(designPoints)
    N = designPoints.shape[0]

    #NxNxdimU array with i,jth entry: (u_i,u_j)
    designPointsList = np.tile(designPoints,(N,1)).reshape((N,N,3))
    
    #Transpose just first 2 axes but keep last one as is
    #diff i,jth entry: u_i-u_j
    diff = designPointsList-designPointsList.transpose((1,0,2))
    #dot product on last axis only
    diff2 = np.einsum('ijk,ijk -> ij',diff,diff)
    kernelStar = np.exp(-diff2)
    kernelStarInverse = np.linalg.inv(kernelStar)
    kernelStarInverseDotPhiStar = kernelStarInverse @ phiStar
    # K_*(u) = np.exp(np.einsum('ij,ij->i',u-designPoints,u-designPoints))
    mean = lambda u : np.exp(-np.einsum('ij,ij->i',u-designPoints,u-designPoints)) @ kernelStarInverseDotPhiStar   
    #At the moment just doing the simple case for \Phi_N(u) = mean(u)
    return mean

def GaussianEmulator(phi, kernel, designPoints):
    """Creates a Guassian Emulator with a given kernel.  SLOW since has loops
    Currently just sets the Guassian Emulator to be the mean"""
    
    phiStar = phi(designPoints)
    N = designPoints.shape[0]
    
    #Create K_*:
#    code for a general kernel:  SLOW since not vectorzed
    kernelStar = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            kernelStar[i,j] = kernel(designPoints[i], designPoints[j])
    kernelStarInverse = np.linalg.inv(kernelStar)
    kernelStarInverseDotPhiStar = kernelStarInverse @ phiStar
    def mean(u):
        k_star = np.zeros(N)
        for i in range(N):
            k_star[i] = kernel(u,designPoints[i])
        
        return k_star @ kernelStarInverseDotPhiStar   
     
    #At the moment just doing the simple case for \Phi_N(u) = mean(u)
    return mean   

def normalizeVector(a):
    """Normalises the vector 'a' but keeps 0 vectors as 0 vectors"""
    norm = np.sum(a*a, 1)
    for x in np.nditer(norm,op_flags=['readwrite']):
        if(x == 0):
            x[...] = 1
    return a/(np.sqrt(norm)[:,None])

#%%Test case with easy phi.  G = identity
#First neeed to generate some data y
#setup
sigma = 1
dimU = 3
length = 10**5 #length of random walk in MCMC
numberDesignPoints = 10 #in each dimension
speedRandomWalk = 0.5
#End points of n-dim lattice for the design points
minRange = -2
maxRange = 2

#Generate data
u = np.random.randn(dimU)
y = u + sigma*np.random.standard_normal(u.shape)

#phi(u) = |y-u|/(2sigma)
phi = lambda u : math.sqrt(np.dot(y-u,y-u))/(2*sigma)
v1phi = np.vectorize(lambda u: np.linalg.norm(y-u)/(2*sigma))
v2phi = np.vectorize(lambda u: np.linalg.norm(y-u)/(2*sigma), signature='(i)->()')

#Create a kernel, just e^(-l^2 norm).
kernel = lambda x,y: np.exp(-np.dot(x-y,x-y))

#List of N^dimU design points in grid
if dimU > 1:
    designPointsGrid = np.meshgrid(*[np.linspace(minRange,maxRange,numberDesignPoints) for _ in range(dimU)])
    designPoints = np.hstack(designPointsGrid).swapaxes(0,1).reshape(dimU,-1).T
    #or np.meshgrid(*[np.linspace(i,j,numPoints)[:-1] for i,j in zip(mins,maxs)])
    GP = GaussianEmulator_Gauss(v2phi, designPoints)
else:
    designPoints = np.linspace(minRange, maxRange, numberDesignPoints)
    GP = GaussianEmulator_Gauss(v1phi, designPoints)



#%% Calculations
densityPrior = lambda u: normalDensity(u)*np.exp(-phi(u))
densityPost = lambda u: normalDensity(u)*np.exp(-GP(u))

x0 = np.zeros(dimU)
print('Running MCMC with length:', length)
#t0 = time.clock()
distPrior = MHRandomWalk(densityPrior, length, x0=x0, speed=speedRandomWalk)
#t1 = time.clock()
#print('CPU time calculating distPrior:', t1-t0)
#cProfile.runctx('MHRandomWalk(densityPost, length, x0=x0, speed=speedRandomWalk)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)

#t0 = time.clock()
distPost = MHRandomWalk(densityPost, length, x0=x0, speed=speedRandomWalk)
#t1 = time.clock()
#print('CPU time calculating distPost:', t1-t0)



#%% Plotting
#Plotting phi and GP of phi:
plotFlag = 0
if plotFlag == 1:      
    #Vectorise GP for plotting
    vGP = np.vectorize(lambda u: GP(u), signature='(i)->()')
    if dimU == 1:
        #This is just 1d plot
        t = np.linspace(-2,2,20)
        plt.plot(t,vGP(t))
        #plt.plot(t,v2phi(t), color='green')
        plt.plot(designPoints,v1phi(designPoints), 'ro')
        plt.show()
    elif dimU == 2:
        X = np.linspace(minRange, maxRange, 20)
        Y = np.linspace(minRange, maxRange, 20)
        Z = np.hstack(np.meshgrid(X, Y)).swapaxes(0,1).reshape(2,-1).T
        X = Z[:,0]
        Y = Z[:,1]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #Plot the surface
        ax.plot_trisurf(X, Y, vGP(Z), antialiased=True)
        #Plot the design points
        ax.scatter(designPoints[:,0], designPoints[:,1], v2phi(designPoints), color='green')
        plt.show()
        
#%% Plot hist        
if plotFlag:
    plt.figure()
    if dimU == 1:
        plt.hist(distPrior, bins=77, alpha=0.5, density=True, label='Prior')
        plt.hist(distPost, bins=77, alpha=0.5, density=True, label='Post')
        plt.legend(loc='upper right')
        plt.show()
    elif dimU == 2:
        plt.hist(distPrior[:,0], bins=77, alpha=0.5, density=True, label='Prior')
        plt.hist(distPost[:,0], bins=77, alpha=0.5, density=True, label='Post')
        plt.legend(loc='upper right')
        plt.show()

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