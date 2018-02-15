"""
Solving a PDE inverse problem using MCMC
Trick: uses a Gaussian Process to approximate the solution to the given PDE
Speeds up each evaulation in MCMC
PDE solved at design points using FEM

Tricky bit: don't know the points will call GP beforehand so have to generate the GP 'on the run'

PDE in p is:
-d/dx (k(x; u) dp/dx(x; u)) = 1 in (0,1)
p(0; u) = 0
p(1; u) = 10

k(x; u) = 1/100 + \sum_j^d u_j/(200(d + 1)) * sin(2\pi jx),
where u \in [-1,1]^d and the truth u^* is randomly generated."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import timeit
#import cProfile
#import pstats

from GaussianProcess import GaussianProcess as gp
import PDE_A as PDE
from MCMC import runMCMC

#%% Setup variables and functions
sigma = np.sqrt(10 ** -3) #size of the noise in observations
dim_U = 3
length = 10 ** 5 #length of MCMC
num_design_points = 20 #in each dimension
speed_random_walk = 0.1
#End points of n-dim lattice for the design points
min_range = -1
max_range = 1
num_obs = 10

#N: number basis functions for solving PDE
N = 2 ** 10
#point to solve PDE at
x = 0.4

#Generate data
#The truth u_dagger lives in [-1,1]
u_dagger = 2*np.random.rand(dim_U) - 1
G_u_dagger = PDE.solve_at_x(u_dagger, N, x)
y = np.broadcast_to(G_u_dagger, (num_obs, dim_U)) + sigma*np.random.standard_normal((num_obs, dim_U))

#uniform density for |x[i]| < 1
uniform_density = lambda x: 1*(np.amax(np.abs(x)) <= 1)

phi = lambda u: np.sum((y - PDE.solve_at_x(u,N,x)) ** 2)/(2*sigma)
vphi = np.vectorize(phi, signature='(i)->()')

#Create Gaussian Process with exp kernel
design_points = gp.create_uniform_grid(min_range, max_range, num_design_points, dim_U)
GP = gp(design_points, vphi(design_points))
    
#Grid points to interpolate with
num_interp_points = 4 * num_design_points

#%% Calculations
#u lives in [-1,1] so use uniform dist as prior or could use normal with cutoff |x| < 2 
density_prior = uniform_density

def MCMC_helper(density_post, name):
    return runMCMC(density_post, length, speed_random_walk, x0, x, N, name, PDE)

flag_run_MCMC = 1
if flag_run_MCMC:
    x0 = np.zeros(dim_U)
    print('Parameter is:', u_dagger)
    print('Solution to PDE at',x,'for true parameter is:', G_u_dagger)
    print('Mean of y for', num_obs,'observations is:', np.sum(y)/(num_obs*dim_U))
    
    if 0:
        density_post = lambda u: np.exp(-phi(u))*density_prior(u)
        name = 'True posterior'
        run_true = MCMC_helper(density_post, name)
    
    if 0:
        density_post = lambda u: np.exp(-GP.mean(u))*density_prior(u)
        name = 'GP as mean - ie marginal approximation'
        run_mean = MCMC_helper(density_post, name)
        
    if 1:
        density_post = lambda u: np.exp(-GP.GP_eval(u))*density_prior(u)
        name = 'GP - one evaluation'
        run_GP = MCMC_helper(density_post, name)
    
    if 0:
        interp = GP.GP(num_interp_points)
        density_post = lambda u: np.exp(-interp(u))*density_prior(u)
        name = 'pi^N_rand via interpolation'
        run_rand = MCMC_helper(density_post, name)

#%% Debugging section:
#Plotting phi and GP of phi:
flag_plot = 0
if flag_plot:      
    #Vectorise GP for plotting
    vGP = np.vectorize(lambda u: GP.GP_eval(u), signature='(i)->()')
    if dim_U == 1:
        #This is just 1d plot
        t = np.linspace(min_range, max_range, 20)
        vGP = np.vectorize(lambda u: GP.GP_eval(u))
        plt.plot(t, vGP(t))
        #plt.plot(t,v2phi(t), color='green')
        plt.plot(design_points, vphi(design_points), 'ro')
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
        ax.plot_trisurf(X, Y, vGP(Z))
        #Plot the design points
        ax.scatter(design_points[:,0], design_points[:,1], vphi(design_points), color='green')
        plt.show()
        
#%% Plot Likelihood:
#likelihood for debugging and checking problems
flag_plot = 0
if flag_plot:
    plt.figure()
    X = np.linspace(-2,2,40)
    v_likelihood = np.vectorize(lambda u,y: np.exp(-np.sum((PDE.solve_at_x(y,N,x)-PDE.solve_at_x(u,N,x))**2)/(2*sigma*num_obs)))
    for i in np.linspace(-1,1,5):
        plt.plot(X,v_likelihood(i,X),label=i)
    plt.legend(loc='upper right')
    plt.title('Likelihood for different truths u_dagger' + ' at x=' + str(x))
    plt.show()

#likelihood for u_dagger in 1d
flag_plot = 0
if flag_plot:
    plt.figure()
    X = np.linspace(-2,2,40)
    v_likelihood = np.vectorize(lambda u,y: np.exp(-np.sum((PDE.solve_at_x(y,N,x)-PDE.solve_at_x(u,N,x))**2)/(2*sigma*num_obs)))
    plt.plot(X,v_likelihood(u_dagger,X),label=str(u_dagger))
    plt.legend(loc='upper right')
    plt.title('Likelihood for different truth u_dagger' + str(u_dagger) + ' at x=' + str(x))
    plt.show()
        
#%% Plot solution to PDE at different parameters
flag_plot = 0
if flag_plot:
    plt.figure()
    X = np.linspace(-1,1,40)
    vPDE_at_x = np.vectorize(lambda u: PDE.solve_at_x(u, N, x))
    plt.plot(X,vPDE_at_x(X))
    plt.title('Solution to PDE for different parameters')
    plt.show()