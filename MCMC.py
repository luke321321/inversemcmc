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
#kv is modified Bessel function of second kind
from scipy.special import gamma, kv
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
#import time, timeit
#import cProfile, pstats

#We pass dimension via x0
def MHRandomWalk(density, length, speed=0.5, x0=np.array([0]), burnTime=200):
    #Calculate dim of parameter space
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


def normalizeVector(a):
    """Normalises the vector 'a' but keeps 0 vectors as 0 vectors"""
    norm = np.sum(a*a, 1)
    for x in np.nditer(norm,op_flags=['readwrite']):
        if(x == 0):
            x[...] = 1
    return a/(np.sqrt(norm)[:,None])

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

def GaussianEmulator_exp(phi, designPoints):
    """Creates a Guassian Emulator with the kernel being a Exponential.
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
    kernelStar = np.exp(-np.sqrt(diff2))
    kernelStarInverse = np.linalg.inv(kernelStar)
    kernelStarInverseDotPhiStar = kernelStarInverse @ phiStar
    # K_*(u) = np.exp(np.einsum('ij,ij->i',u-designPoints,u-designPoints))
    mean = lambda u : np.exp(-np.sqrt(np.einsum('ij,ij->i',u-designPoints,u-designPoints))) @ kernelStarInverseDotPhiStar   
    #At the moment just doing the simple case for \Phi_N(u) = mean(u)
    return mean

def GaussianEmulator_Matern(phi, designPoints, nu = 0, sig2 = 1, lam = 1):
    """Creates a Guassian Emulator with the kernel being a Matern Kernal.
    Currently just sets the Guassian Emulator to be the mean
    
    Defaults: to Gaussian kernal
    Gaussian: nu = 0
    Exp: nu = 1/2
        
    """
    phiStar = phi(designPoints)
    N = designPoints.shape[0]

    #NxNxdimU array with i,jth entry: (u_i,u_j)
    designPointsList = np.tile(designPoints,(N,1)).reshape((N,N,3))
    
    #Transpose just first 2 axes but keep last one as is
    #diff i,jth entry: u_i-u_j
    diff = designPointsList-designPointsList.transpose((1,0,2))
    #dot product on last axis only
    r2 = np.einsum('ijk,ijk -> ij',diff,diff)
    
    if nu == 0:
        kernalStar = sig2*np.exp(-r2/lam)
    elif nu == 0.5:
        kernalStar = sig2*np.exp(-np.sqrt(r2)/lam)
    else:
        rt2nu = math.sqrt(2*nu)/lam
        const = (sig2/gamma(nu)*math.pow(2,nu-1))
        kernalStar = const*np.power(rt2nu*r2,nu)*kv(nu,rt2nu*r2)
        #if r2[i,j] = 0 then want kernalStar[i,j] = 1
        #Asymptotics when r2 = 0 not the best
        where_NaNs = np.isnan(kernalStar)
        kernalStar[where_NaNs] = 1
        
    kernelStarInverse = np.linalg.inv(kernalStar)
    kernelStarInverseDotPhiStar = kernelStarInverse @ phiStar
    
    #k_*(u) = kernal(u,u_i), u_i are design points
    if nu == 0:
        mean = lambda u : np.exp(-np.einsum('ij,ij->i',u-designPoints,u-designPoints)) @ kernelStarInverseDotPhiStar
    elif nu == 0.5:
        mean = lambda u : np.exp(-np.sqrt(np.einsum('ij,ij->i',u-designPoints,u-designPoints))) @ kernelStarInverseDotPhiStar
    else:
        def mean(u):
            #Asymptotics when r2 = 0 not the best
            r2 = np.einsum('ij,ij->i',u-designPoints,u-designPoints)
            rt2nu = math.sqrt(2*nu)/lam
            const = (sig2/gamma(nu)*math.pow(2,nu-1))
            k_star = const*np.power(rt2nu*r2,nu)*kv(nu,rt2nu*r2)
            #if r2[i,j] = 0 then want k_star[i,j] = 1
            where_NaNs = np.isnan(k_star)
            k_star[where_NaNs] = 1
            return k_star @ kernelStarInverseDotPhiStar
        
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

#%%
def solvePDE(u,N):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points"""
    
    #create N Chebyshev nodes in (0,1)
    nodes = np.zeros(N+2)
    nodes[1:-1] = np.cos((2*np.arange(N)+1)*np.pi/(2*N) - np.pi)/2 + 0.5
    nodes[-1] = 1
    
    A = stiffnessMatrix(nodes,u)
    b = calB(nodes)
    
    #solve the PDE
    p = spsolve(A,b)
    return p, nodes

def stiffnessMatrix(nodes,u):
    """Returns the sparse stiffness matrix from nodes and k(x;u), nodes include enpoints"""
    #To speed up could just call integralK once and then splice that
    
    #calculate derivative of basis functions - which is just a step function
    vL = 1/(nodes[1:-1] - nodes[:-2])
    vR = 1/(nodes[2:] - nodes[1:-1])
    
    #integrate K between the nodes
    intK = integralK(nodes[:-1],nodes[1:],u)
    
    #Construct the stiffness matrix
    diag0 = (vL**2)*intK[:-1] + (vR**2)*intK[1:]
    diag1 = -vR[:-1]*vL[1:]*intK[1:-1]
    A = diags([diag1,diag0,diag1],[-1,0,1], format="csr")
    return A

def calB(nodes):
    """Returns the vector b to solve the PDE"""
    return (nodes[2:] - nodes[:-2])/2

def integralK(a,b,u):
    """Returns the integral of k(x;u) from a to b, both 1d arrays of the same dimension"""
    dimU = u.size
    
    #tile arrays to 2d arrays
    A = np.broadcast_to(a,(dimU,a.size))
    B = np.broadcast_to(b,(dimU,b.size))
    J = np.broadcast_to(np.arange(dimU)+1,(a.size,dimU)).T
    U = np.broadcast_to(u,(a.size,dimU)).T
    
    #calculate k(x;u) at nodes x
    toSum = U*(np.cos(2*np.pi*J*A) - np.cos(2*np.pi*J*B))/(2*np.pi*J)
    return (b-a)/100 + np.sum(toSum, axis=0)/(200*(dimU + 1))

##Testing code for PDE solver
#u = np.random.randn(20)
#N = 100000
#p,nodes = solvePDE(u,N)
#plt.plot(nodes[1:-1],p)
#cProfile.runctx('integralK(nodes[:-1],nodes[1:],u)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)

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
    #0 is Gaussian kernal
    GP = GaussianEmulator_Matern(v2phi, designPoints, 0)
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
plotFlag = 1
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
    else:
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