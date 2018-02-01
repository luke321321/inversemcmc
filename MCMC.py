"""
Created on Wed Dec  6 14:33:52 2017

@author: s1002685

"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#kv is modified Bessel function of second kind
from scipy.special import gamma, kv
from scipy.sparse import diags
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator, interp1d
#from scipy.sparse.linalg import spsolve
#Quicker sparse LA solver:
#install via: 'conda install -c haasad pypardiso'
from pypardiso import spsolve
#Progress bar
from tqdm import tqdm

import time
import timeit
import cProfile
import pstats

#We pass dimension via x0
def MH_random_walk(density, length, speed=0.5, x0=np.array([0]), burn_time=1000):
    #Calculate dim of parameter space
    if isinstance(x0, np.ndarray):
        dim = x0.size
    else:
        dim = 1
        
    x = np.zeros((burn_time + length, dim))
    #Generates normal rv in R^n of length=speed or 0.
    rvNormal = speed*np.random.randn(burn_time + length, dim)
    
    accepted_count = 0
    
    x[0] = x0
    density_old = density(x0)
    for i in tqdm(range(1,burn_time + length)):
        y = x[i-1] + rvNormal[i]
        if(density_old == 0): #Accept whatever
            x[i] = y
            density_old = density(y)
            if i > burn_time:
                accepted_count += 1
        else:
            u = random.random()
            density_new = density(y)
            if(u <= min(1,density_new/density_old)): #check acceptance
                x[i] = y
                density_old = density_new
                if i > burn_time:
                    accepted_count += 1
            else:
                x[i] = x[i-1]
    return accepted_count, x[burn_time:]


def gaussian_emulator_matern(phi, design_points, nu = np.inf, sig2 = 1, lam = 1):
    """Creates a Guassian Emulator with the kernel being a Matern Kernal.
    Returns functions mean(u), kernal_N(u,v)
    
    Defaults: to Gaussian kernal
    Gaussian: nu = np.inf
    Exp: nu = 1/2
    
    Note for functions mean(u), kernal_N(u,v) 
    If u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector
    and the same for v 
    """
    phiStar = phi(design_points)
    
    #if the design_points are 1d append empty axis for r2_distance method
    if len(design_points.shape) == 1:
        design_points = design_points[:,np.newaxis]
    
    def k(u, v, nu = np.inf, sig2 = 1, lam = 1):
        r2 = r2_distance(u,v)
        
        if nu == np.inf:
            return np.exp(-r2)
        elif nu == 0.5:
            return sig2*np.exp(-np.sqrt(r2)/lam)
        else:
            rt2_nu = math.sqrt(2*nu)/lam
            const = (sig2/gamma(nu)*math.pow(2,nu-1))
            kuv = const*np.power(rt2_nu*r2,nu)*kv(nu,rt2_nu*r2)
            
            #if r2[i,j] = 0 then want kernalStar[i,j] = 1
            #Asymptotics when r2 = 0 not the best
            where_NaNs = np.isnan(kuv)
            kuv[where_NaNs] = 1
            return kuv
    
    kernel_star_inverse = np.linalg.inv(k(design_points,design_points))
    kernel_star_inverse_dot_phi_star = kernel_star_inverse @ phiStar
    
    mean = lambda u : k(u,design_points) @ kernel_star_inverse_dot_phi_star
    kernal_N = lambda u,v: k(u,v) - k(v,design_points) @ kernel_star_inverse @ k(u,design_points).T
    
    return mean, kernal_N

def r2_distance(u,v):
    """Calculates the l^2 distance squared between u and v for u,v being points,
    vectors or list of vectors and returns a point, vector or matrix as appropriate.
    Dimensions of the vectors has to be the same
    
    Note if u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector.
    
    If u.shape = (m x d), v.shape = (n x d) then r2.shape = (m x n)
    If u.shape = (m,)     v.shape = (n x d) then r2.shape = (m x n)
    If u.shape = (m,)     v.shape = (n,)    then r2.shape = (m x n)
    If m or n = 1 then that dimension is squeezed out of r2.shape"""
    
    #First calculate vector/matrix of length of u-v
    dim_U = len(u.shape)
    dim_V = len(v.shape)
    if (dim_U == 1 and dim_V == 1):
        r2 = np.sum(np.square(u-v))
    else:
        V = v[np.newaxis,:]
        U = u[:,np.newaxis]
        if dim_U == 1:
            U = U.T
        diff = U-V
        r2 = np.squeeze(np.einsum('ijk,ijk->ij',diff,diff))
    return r2

#%% PDE solver code
def solve_PDE(u,N):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points"""
    
    #create N Chebyshev nodes in (0,1)
    nodes = np.zeros(N+2)
    nodes[1:-1] = np.cos((2*np.arange(N)+1)*np.pi/(2*N) - np.pi)/2 + 0.5
    nodes[-1] = 1
    
    A = stiffness_matrix(nodes,u)
    b = cal_B(nodes)
    
    #solve the PDE
    p = spsolve(A,b)
    return p, nodes

def stiffness_matrix(nodes,u):
    """Returns the sparse stiffness matrix from nodes and k(x;u), nodes include enpoints"""
    #To speed up could just call integral_K once and then splice that
    
    #calculate derivative of basis functions - which is just a step function
    v_L = 1/(nodes[1:-1] - nodes[:-2])
    v_R = 1/(nodes[2:] - nodes[1:-1])
    
    #integrate K between the nodes
    intK = integral_K(nodes[:-1],nodes[1:],u)
    
    #Construct the stiffness matrix
    diag_0 = (v_L**2)*intK[:-1] + (v_R**2)*intK[1:]
    diag_1 = -v_R[:-1]*v_L[1:]*intK[1:-1]
    A = diags([diag_1,diag_0,diag_1],[-1,0,1], format="csr")
    return A

def cal_B(nodes):
    """Returns the vector b to solve the PDE"""
    return (nodes[2:] - nodes[:-2])/2

def integral_K(a,b,u):
    """Returns the integral of k(x;u) from a to b, both 1d arrays of the same dimension"""
    if isinstance(u, np.ndarray):
        dim_U = u.size
    else:
        dim_U = 1
    
    #tile arrays to 2d arrays
    A = np.broadcast_to(a,(dim_U,a.size))
    B = np.broadcast_to(b,(dim_U,b.size))
    J = np.broadcast_to(np.arange(dim_U)+1,(a.size,dim_U)).T
    U = np.broadcast_to(u,(a.size,dim_U)).T
    
    #calculate k(x;u) at nodes x
    to_sum = U*(np.cos(2*np.pi*J*A) - np.cos(2*np.pi*J*B))/(2*np.pi*J)
    return (b-a)/100 + np.sum(to_sum, axis=0)/(200*(dim_U + 1))

def solve_PDE_at_x(u,N,x):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points, and returns the value of p(x) where 0 < x < 1"""
    p, nodes = solve_PDE(u,N)
    i = np.searchsorted(nodes, x)
    return (p[i-1]*(x - nodes[i-1])+ p[i]*(nodes[i] - x))/(nodes[i] - nodes[i-1])
    
def create_uniform_grid(min_range,max_range,n,dim):
    if dim > 1:
        grid_points_grid = np.meshgrid(*[np.linspace(min_range,max_range,n) for _ in range(dim)])
        grid_points = np.hstack(grid_points_grid).swapaxes(0,1).reshape(dim,-1).T
    else:
        grid_points = np.linspace(min_range, max_range, n)[:,np.newaxis]
    return grid_points

#%% Setup variables and functions
#First neeed to generate some data y
#setup
sigma = 0.1
dim_U = 2
length = 10**4 #length of random walk in MCMC
num_design_points = 5 #in each dimension
speed_random_walk = 0.1
#End points of n-dim lattice for the design points
min_range = -1
max_range = 1
#need above 1000
num_obs = 1
#number basis functions for solving PDE
N = 10 ** 3
#point to solve PDE at
x = 0.45

#Generate data
#The truth u_dagger lives in [-1,0.5]
u_dagger = np.random.rand(dim_U) - 1
#u_dagger = -0.7
G_u_dagger = solve_PDE_at_x(u_dagger,N,x)
y = G_u_dagger
#y = np.broadcast_to(G_u_dagger,(num_obs,dim_U)) + sigma*np.random.standard_normal((num_obs,dim_U))

ROOT2PI = math.sqrt(2*math.pi)
normal_density = lambda x: math.exp(-np.dot(x,x)/2)/ROOT2PI
#uniform density for [-1,1]
uniform_density = lambda x: 1*(np.dot(x,x) <= 2)
#uniform_density = lambda x: 1*((np.dot(x,x) <= 2) & (x-0.5 <= 0).all())
normal_density2 = lambda x: math.exp(-np.dot(x,x)/8)*(np.dot(x,x) <= 4)

phi = lambda u: np.sum((y-solve_PDE_at_x(u,N,x))**2)/(2*sigma*num_obs)

#Create Gaussian Process with exp kernel
design_points = create_uniform_grid(min_range,max_range,num_design_points, dim_U)
if dim_U > 1:
    vphi = np.vectorize(lambda u: np.linalg.norm(y-solve_PDE_at_x(u,N,x))**2/(2*sigma*num_obs))
else:
    vphi = np.vectorize(lambda u: np.linalg.norm(y-solve_PDE_at_x(u,N,x))**2/(2*sigma*num_obs), signature='(i)->()')
GP_mean, GP_kernel = gaussian_emulator_matern(vphi, design_points, 1/2)
    
piN_rand = lambda u: np.exp(-GP_mean(u))*normal_density2(u)

#%% Generate \phi_N
num_GP_grid_points = 50
num_GP_realisations = 100
GP_grid_points = create_uniform_grid(min_range, max_range, num_GP_grid_points, dim_U)
#create Gaussian Process - generate random variables by calling GP_rv.rvs()
GP_rv = multivariate_normal(GP_mean(GP_grid_points), GP_kernel(GP_grid_points,GP_grid_points), allow_singular=True)

if dim_U == 1:
    phiN_marginal = interp1d(GP_grid_points.squeeze(), GP_rv.rvs(),
                                            copy=False, assume_sorted=True)
else:
    phiN_marginal = RegularGridInterpolator(GP_grid_points.squeeze(),GP_rv.rvs())
piN_marginal = lambda u: np.exp(-phiN_marginal(u))*normal_density2(u)

#TODO piN_marginal throws and error if interpolation is outside of range - is it enough for it to be nan?
# Or do we interpolate over longer range?

#TODO check n-dim case!

#%% Calculations
#u lives in [-1,1] so use uniform dist as prior
#OR can use normal_density2, which is normal with cutoff |x| < 2 
density_prior = lambda u: uniform_density(u)*np.exp(-phi(u))
density_post = lambda u: uniform_density(u)*np.exp(-GP_mean(u))


#Testing generating Gaussian Process
n = 100
X_test = np.linspace(-1, 1, n).reshape(-1,1)
f_post = np.random.multivariate_normal(GP_mean(X_test), GP_kernel(X_test,X_test), size=50).T
plt.figure()
plt.plot(X_test, f_post)
plt.plot(design_points,vphi(design_points), 'ro')
plt.title('50 sample from the GP posterior')
#plt.axis([-1, 1, -3, 3])
plt.show()

flag_run_MCMC = 0
if flag_run_MCMC:
    x0 = np.zeros(dim_U)
    print('Parameter is:', u_dagger)
    print('Solution to PDE at',x,'for true parameter is:', G_u_dagger)
    print('Mean of', num_obs,'observations is:', np.sum(y,0)/num_obs)
    print('Running MCMC with length:', length, 'and speed:', speed_random_walk)
#    t0 = time.clock()
    accepts, dist_prior = MH_random_walk(density_prior, length, x0=x0, speed=speed_random_walk)
#    t1 = time.clock()
#    print('CPU time calculating dist_prior:', t1-t0)
    #cProfile.runctx('MH_random_walk(density_prior, length, x0=x0, speed=speed_random_walk)'
    #                , globals(), locals(), '.prof')
    #s = pstats.Stats('.prof')
    #s.strip_dirs().sort_stats('time').print_stats(30)
    mean_dist_prior = np.sum(dist_prior,0)/length
    print('Mean of dist_prior is:', mean_dist_prior)
    print('We accepted this number of times:', accepts)
    sol_at_mean = solve_PDE_at_x(mean_dist_prior,N,x)
    print('Solution to PDE at mean of dist_prior is:', sol_at_mean)
    
    #t0 = time.clock()
    #distPost = MH_random_walk(density_post, length, x0=x0, speed=speed_random_walk)
    #t1 = time.clock()
    #print('CPU time calculating distPost:', t1-t0)



#%% Plotting
#Plotting phi and GP of phi:
flag_plot = 1
if flag_plot:      
    #Vectorise GP for plotting
    vGP = np.vectorize(lambda u: GP_mean(u), signature='(i)->()')
    if dim_U == 1:
        #This is just 1d plot
        t = np.linspace(min_range,max_range,20)
        vGP = np.vectorize(lambda u: GP_mean(u))
        plt.plot(t,vGP(t))
        #plt.plot(t,v2phi(t), color='green')
        plt.plot(design_points,vphi(design_points), 'ro')
        plt.show()
    elif dim_U == 2:
        X = np.linspace(min_range, max_range, 20)
        Y = np.linspace(min_range, max_range, 20)
        Z = np.hstack(np.meshgrid(X, Y)).swapaxes(0,1).reshape(2,-1).T
        X = Z[:,0]
        Y = Z[:,1]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #Plot the surface
        ax.plot_trisurf(X, Y, vGP(Z), antialiased=True)
        #Plot the design points
        ax.scatter(design_points[:,0], design_points[:,1], v2phi(design_points), color='green')
        plt.show()
        
#%% Plot hist  
flag_plot = 0
if flag_plot:
    plt.figure()
    if dim_U == 1:
        plt.hist(dist_prior, bins=101, alpha=0.5, density=True, label='Prior')
#        plt.hist(distPost, bins=101, alpha=0.5, density=True, label='Post')
    else:
        plt.hist(dist_prior[:,0], bins=101, alpha=0.5, density=True, label='Prior')
#        plt.hist(distPost[:,0], bins=101, alpha=0.5, density=True, label='Post')
    plt.legend(loc='upper right')
    plt.show()
        
#%% Plot Likelihood:
#likelihood for debugging and checking problems
flag_plot = 0
if flag_plot:
    plt.figure()
    X = np.linspace(-2,2,40)
    v_likelihood = np.vectorize(lambda u,y: np.exp(-np.sum((solve_PDE_at_x(y,N,x)-solve_PDE_at_x(u,N,x))**2)/(2*sigma*num_obs)))
    for i in np.linspace(-1,1,5):
        plt.plot(X,v_likelihood(i,X),label=i)
    plt.legend(loc='upper right')
    plt.title('Likelihood for different truths u_dagger')
    plt.show()

#likelihood for u_dagger
flag_plot = 0
if flag_plot:
    plt.figure()
    X = np.linspace(-2,2,40)
    v_likelihood = np.vectorize(lambda u,y: np.exp(-np.sum((solve_PDE_at_x(y,N,x)-solve_PDE_at_x(u,N,x))**2)/(2*sigma*num_obs)))
    plt.plot(X,v_likelihood(u_dagger,X),label=str(u_dagger))
    plt.legend(loc='upper right')
    plt.title('Likelihood for different truth u_dagger' + str(u_dagger))
    plt.show()
        
#%% Plot solution to PDE at different parameters
flag_plot = 0
if flag_plot:
    plt.figure()
    X = np.linspace(-2,2,40)
    vPDE_at_x = np.vectorize(lambda u: solve_PDE_at_x(u,N,x))
    plt.plot(X,vPDE_at_x(X))
    plt.title('Solution to PDE for different parameters')
    plt.show()

#%% Testing Metroplis-Hastings algorithm
#t0 = time.clock()
#x = MH_random_walk(normal_density, length=100000, speed=0.5, x0=np.array([0,0,0,0,0]))
#t1 = time.clock()
#print('CPU time for loops in Python', t1-t0)
#plt.hist(x, bins=77,  density=True)
#plt.show()
    
#%%Testing code for PDE solver
#u = np.random.randn(1)
#N = 100000
#x = 0.45
#p, nodes = solve_PDE(u,N)
#plt.plot(nodes[1:-1],p)
#print('Value at 0.5:', solve_PDE_at_x(u,N,0.5))
#cProfile.runctx('solve_PDE_at_x(u,N,x)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)