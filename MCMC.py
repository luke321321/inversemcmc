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
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator, interp1d
#from scipy.sparse.linalg import spsolve
#Quicker sparse LA solver:
from pypardiso import spsolve
#Progress bar
from tqdm import tqdm

import time, timeit
import cProfile, pstats

#We pass dimension via x0
def MHRandomWalk(density, length, speed=0.5, x0=np.array([0]), burnTime=1000):
    #Calculate dim of parameter space
    if isinstance(x0, np.ndarray):
        dim = x0.size
    else:
        dim = 1
        
    x = np.zeros((burnTime + length, dim))
    #Generates normal rv in R^n of length=speed or 0.
    rvNormal = speed*np.random.randn(burnTime + length, dim)
    
    acceptNumbers = 0
    
    x[0] = x0
    densityOld = density(x0)
    for i in tqdm(range(1,burnTime + length)):
        y = x[i-1] + rvNormal[i]
        if(densityOld == 0): #Accept whatever
            x[i] = y
            densityOld = density(y)
            if i > burnTime:
                acceptNumbers += 1
        else:
            u = random.random()
            densityNew = density(y)
            if(u <= min(1,densityNew/densityOld)): #check acceptance
                x[i] = y
                densityOld = densityNew
                if i > burnTime:
                    acceptNumbers += 1
            else:
                x[i] = x[i-1]
    return acceptNumbers, x[200:]


def GaussianEmulator_Matern(phi, designPoints, nu = np.inf, sig2 = 1, lam = 1):
    """Creates a Guassian Emulator with the kernel being a Matern Kernal.
    Returns functions mean(u), kernalN(u,v)
    
    Defaults: to Gaussian kernal
    Gaussian: nu = np.inf
    Exp: nu = 1/2
    
    Note for functions mean(u), kernalN(u,v) 
    If u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector
    and the same for v 
    """
    phiStar = phi(designPoints)
    
    #if the designPoints are 1d append empty axis for r2Distance method
    if len(designPoints.shape) == 1:
        designPoints = designPoints[:,np.newaxis]
    
    def k(u, v, nu = np.inf, sig2 = 1, lam = 1):
        r2 = r2Distance(u,v)
        
        if nu == np.inf:
            return np.exp(-r2)
        elif nu == 0.5:
            return sig2*np.exp(-np.sqrt(r2)/lam)
        else:
            rt2nu = math.sqrt(2*nu)/lam
            const = (sig2/gamma(nu)*math.pow(2,nu-1))
            kuv = const*np.power(rt2nu*r2,nu)*kv(nu,rt2nu*r2)
            
            #if r2[i,j] = 0 then want kernalStar[i,j] = 1
            #Asymptotics when r2 = 0 not the best
            where_NaNs = np.isnan(kuv)
            kuv[where_NaNs] = 1
            return kuv
    
    kernelStarInverse = np.linalg.inv(k(designPoints,designPoints))
    kernelStarInverseDotPhiStar = kernelStarInverse @ phiStar
    
    mean = lambda u : k(u,designPoints) @ kernelStarInverseDotPhiStar
    kernalN = lambda u,v: k(u,v) - k(v,designPoints) @ kernelStarInverse @ k(u,designPoints).T
    
    return mean, kernalN

def r2Distance(u,v):
    """Calculates the l^2 distance squared between u and v for u,v being points,
    vectors or list of vectors and returns a point, vector or matrix as appropriate.
    Dimensions of the vectors has to be the same
    
    Note if u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector.
    
    If u.shape = (m x d), v.shape = (n x d) then r2.shape = (m x n)
    If u.shape = (m,)     v.shape = (n x d) then r2.shape = (m x n)
    If u.shape = (m,)     v.shape = (n,)    then r2.shape = (m x n)
    If m or n = 1 then that dimension is squeezed out of r2.shape"""
    
    #First calculate vector/matrix of length of u-v
    dimU = len(u.shape)
    dimV = len(v.shape)
    if (dimU == 1 and dimV == 1):
        r2 = np.sum(np.square(u-v))
    else:
        V = v[np.newaxis,:]
        U = u[:,np.newaxis]
        if dimU == 1:
            U = U.T
        diff = U-V
        r2 = np.squeeze(np.einsum('ijk,ijk->ij',diff,diff))
    return r2

#%% PDE solver code
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
    if isinstance(u, np.ndarray):
        dimU = u.size
    else:
        dimU = 1
    
    #tile arrays to 2d arrays
    A = np.broadcast_to(a,(dimU,a.size))
    B = np.broadcast_to(b,(dimU,b.size))
    J = np.broadcast_to(np.arange(dimU)+1,(a.size,dimU)).T
    U = np.broadcast_to(u,(a.size,dimU)).T
    
    #calculate k(x;u) at nodes x
    toSum = U*(np.cos(2*np.pi*J*A) - np.cos(2*np.pi*J*B))/(2*np.pi*J)
    return (b-a)/100 + np.sum(toSum, axis=0)/(200*(dimU + 1))

def solvePDEatx(u,N,x):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points, and returns the value of p(x) where 0 < x < 1"""
    p, nodes = solvePDE(u,N)
    i = np.searchsorted(nodes, x)
    return (p[i-1]*(x - nodes[i-1])+ p[i]*(nodes[i] - x))/(nodes[i] - nodes[i-1])

#%%Testing code for PDE solver
#u = np.random.randn(1)
#N = 100000
#x = 0.45
#p,nodes = solvePDE(u,N)
#plt.plot(nodes[1:-1],p)
#print('Value at 0.5:', solvePDEatx(u,N,0.5))
#cProfile.runctx('solvePDEatx(u,N,x)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)
    
def createUniformGrid(minRange,maxRange,n,dim):
    if dim > 1:
        gridPointsGrid = np.meshgrid(*[np.linspace(minRange,maxRange,n) for _ in range(dim)])
        gridPoints = np.hstack(gridPointsGrid).swapaxes(0,1).reshape(dim,-1).T
    else:
        gridPoints = np.linspace(minRange, maxRange, n)[:,np.newaxis]
    return gridPoints

#%% Setup variables and functions
#First neeed to generate some data y
#setup
sigma = 0.1
dimU = 1
length = 10**4 #length of random walk in MCMC
numberDesignPoints = 5 #in each dimension
speedRandomWalk = 0.4
#End points of n-dim lattice for the design points
minRange = -1
maxRange = 1
#need above 1000
numObs = 1
#number basis functions for solving PDE
N=10**3
#point to solve PDE at
x=0.45

#Generate data
#The truth uDagger lives in [-1,0.5]
#uDagger = np.random.rand(dimU) - 1
uDagger = -0.3
GuDagger = solvePDEatx(uDagger,N,x)
y = GuDagger
#y = np.broadcast_to(GuDagger,(numObs,dimU)) + sigma*np.random.standard_normal((numObs,dimU))

root2Pi = math.sqrt(2*math.pi)
normalDensity = lambda x: math.exp(-np.dot(x,x)/2)/root2Pi
#uniform density for [-1,1]
uniformDensity = lambda x: 1*((np.dot(x,x) <= 2) & (x-0.5 <= 0).all())
normalDensity2 = lambda x: math.exp(-np.dot(x,x)/8)*(np.dot(x,x) <= 4)

#phi(u) = |y-u|^2/(2sigma)
#phi = lambda u : math.sqrt(np.dot(y-u,y-u))/(2*sigma)
phi = lambda u: np.sum((y-solvePDEatx(u,N,x))**2)/(2*sigma*numObs)
#phi = lambda u: ((y-u)**2)/2

#Create Gaussian Process
designPoints = createUniformGrid(minRange,maxRange,numberDesignPoints, dimU)
if dimU > 1:
    vphi = np.vectorize(lambda u: np.linalg.norm(y-solvePDEatx(u,N,x))**2/(2*sigma*numObs))
    GP_mean, GP_kernel = GaussianEmulator_Matern(vphi, designPoints, np.inf)
else:
    vphi = np.vectorize(lambda u: np.linalg.norm(y-solvePDEatx(u,N,x))**2/(2*sigma*numObs), signature='(i)->()')
    GP_mean, GP_kernel = GaussianEmulator_Matern(vphi, designPoints, np.inf)
    
piN_rand = lambda u: np.exp(-GP_mean(u))*normalDensity2(u)

#%% Generate \phi_N
numberGPGridPoints = 50
numberGPRealisations = 100
GPGridPoints = createUniformGrid(minRange, maxRange, numberGPGridPoints, dimU)
#create Gaussian Process - generate random variables by calling GP_rv.rvs()
GP_rv = multivariate_normal(GP_mean(GPGridPoints), GP_kernel(GPGridPoints,GPGridPoints), allow_singular=True)

if dimU == 1:
    phiN_marginal = interp1d(GPGridPoints.squeeze(), GP_rv.rvs(),
                                            copy=False, assume_sorted=True)
else:
    phiN_marginal = RegularGridInterpolator(GPGridPoints.squeeze(),GP_rv.rvs())
piN_marginal = lambda u: np.exp(-phiN_marginal(u))*normalDensity2(u)

#TODO piN_marginal throws and error if interpolation is outside of range - is it enough for it to be nan?
# Or do we interpolate over longer range?

#TODO check n-dim case!

#%% Calculations
#u lives in [-1,1] so use uniform dist as prior
densityPrior = lambda u: normalDensity2(u)*np.exp(-phi(u))
densityPost = lambda u: uniformDensity(u)*np.exp(-GP_mean(u))

n = 30
Xtest = np.linspace(-1, 1, n).reshape(-1,1)
f_post = np.random.multivariate_normal(GP_mean(Xtest), GP_kernel(Xtest,Xtest), size=50).T
plt.figure()
plt.plot(Xtest, f_post)
plt.plot(designPoints,vphi(designPoints), 'ro')
plt.title('30 sample from the GP posterior')
#plt.axis([-1, 1, -3, 3])
plt.show()

flagRunMCMC = 0
if flagRunMCMC:
    x0 = np.zeros(dimU)
    print('Parameter is:', uDagger)
    print('Solution to PDE at',x,'for true parameter is:', GuDagger)
    print('Mean of', numObs,'observations is:', np.sum(y,0)/numObs)
    print('Running MCMC with length:', length)
    t0 = time.clock()
    accepts, distPrior = MHRandomWalk(densityPrior, length, x0=x0, speed=speedRandomWalk)
    t1 = time.clock()
    print('CPU time calculating distPrior:', t1-t0)
    #cProfile.runctx('MHRandomWalk(densityPrior, length, x0=x0, speed=speedRandomWalk)'
    #                , globals(), locals(), '.prof')
    #s = pstats.Stats('.prof')
    #s.strip_dirs().sort_stats('time').print_stats(30)
    print('Mean of distPrior is:', np.sum(distPrior,0)/length)
    print('We accepted this number of times:', accepts)
    
    #t0 = time.clock()
    #distPost = MHRandomWalk(densityPost, length, x0=x0, speed=speedRandomWalk)
    #t1 = time.clock()
    #print('CPU time calculating distPost:', t1-t0)



#%% Plotting
#Plotting phi and GP of phi:
flagPlot = 0
if flagPlot:      
    #Vectorise GP for plotting
    vGP = np.vectorize(lambda u: GP_mean(u), signature='(i)->()')
    if dimU == 1:
        #This is just 1d plot
        t = np.linspace(minRange,maxRange,20)
        vGP = np.vectorize(lambda u: GP_mean(u))
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
flagPlot = 0
if flagPlot:
    plt.figure()
    if dimU == 1:
        plt.hist(distPrior, bins=101, alpha=0.5, density=True, label='Prior')
#        plt.hist(distPost, bins=101, alpha=0.5, density=True, label='Post')
        plt.legend(loc='upper right')
        plt.show()
    else:
        plt.hist(distPrior[:,0], bins=101, alpha=0.5, density=True, label='Prior')
#        plt.hist(distPost[:,0], bins=101, alpha=0.5, density=True, label='Post')
        plt.legend(loc='upper right')
        plt.show()
        
#%% Plot Likelihood:
#likelihood for debugging and checking problems
flagPlot = 0
if flagPlot:
    plt.figure()
    X = np.linspace(-2,2,40)
    vlikelihood = np.vectorize(lambda u,y: np.exp(-np.sum((solvePDEatx(y,N,x)-solvePDEatx(u,N,x))**2)/(2*sigma*numObs)))
    for i in np.linspace(-1,1,5):
        plt.plot(X,vlikelihood(i,X),label=i)
        
#%% Plot solution to PDE at different parameters
flagPlot = 0
if flagPlot:
    plt.figure()
    X = np.linspace(-2,2,40)
    vPDEatx = np.vectorize(lambda u: solvePDEatx(u,N,x))
    plt.plot(X,vPDEatx(X))

#%% Testing Metroplis-Hastings algorithm
#t0 = time.clock()
#x = MHRandomWalk(normalDensity, length=100000, speed=0.5, x0=np.array([0,0,0,0,0]))
#t1 = time.clock()
#print('CPU time for loops in Python', t1-t0)
#plt.hist(x, bins=77,  density=True)
#plt.show()

#%% Testing normaliseVector
#print('Checking normaliseVector code')
#a = np.array([[0,1],[0,0],[2,3]])
#b = normalizeVector(a)
#ans = np.array([[0,1],[0,0],[2/math.sqrt(13), 3/math.sqrt(13)]])
#print('normaliseVector, Test 1:', np.allclose(b,ans))