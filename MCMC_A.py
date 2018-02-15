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
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Progress bar
from tqdm import tqdm

#import timeit
#import cProfile
#import pstats

from GaussianProcess import GaussianProcess as gp
import PDE_A as PDE

#We pass dimension via x0
def MH_random_walk(density, length, speed=0.5, x0=np.array([0]), burn_time=1000):
    #Calculate dim of parameter space
    if isinstance(x0, np.ndarray):
        dim = x0.size
    else:
        dim = 1
        
    x = np.zeros((burn_time + length, dim))
    #Pre-generates the normal rv in R^n
    rvNormal = speed*np.random.normal(size=(burn_time + length, dim))
    
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

def runMCMC(dens, length, speed_random_walk, x0, x, N, name):
    """Helper function to start off running MCMC"""
    print('\n' + name)
    print('Running MCMC with length:', length, 'and speed:', speed_random_walk)
    accepts, run = MH_random_walk(dens, length, x0=x0, speed=speed_random_walk)

    mean = np.sum(run, 0)/length
    print('Mean is:', mean)
    print('We accepted this number of times:', accepts)
    sol_at_mean = PDE.solve_at_x(mean,N,x)
    print('Solution to PDE at mean is:', sol_at_mean)
    plot_dist(run, name)
    return run

def plot_dist(dist, title):
    """Plots the distribution on a grid"""
    sns.set(color_codes=True)
    sns.set_style('white')
    sns.set_style('ticks')
    g = sns.PairGrid(pd.DataFrame(dist), despine=True)
    g.map_diag(sns.kdeplot, legend=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d", n_levels=4)
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

#%% Setup variables and functions
sigma = 0.05 #size of the noise in observations
dim_U = 3
length = 2 ** 10 #length of MCMC
num_design_points = 20 #in each dimension
speed_random_walk = 0.1
#End points of n-dim lattice for the design points
min_range = -1
max_range = 1
num_obs = 25

#N: number basis functions for solving PDE
N = 2 ** 12
#point to solve PDE at
x = 0.4

#Generate data
#The truth u_dagger lives in [-1,1]
u_dagger = 2*np.random.rand(dim_U) - 1
G_u_dagger = PDE.solve_at_x(u_dagger, N, x)
y = np.broadcast_to(G_u_dagger, (num_obs, dim_U)) + sigma*np.random.standard_normal((num_obs, dim_U))

#uniform density for |x[i]| < 1
uniform_density = lambda x: 1*(np.amax(np.abs(x)) <= 1)

phi = lambda u: np.sum((y - PDE.solve_at_x(u,N,x)) ** 2)/(2*sigma*num_obs)
vphi = np.vectorize(phi, signature='(i)->()')

#Create Gaussian Process with exp kernel
design_points = gp.create_uniform_grid(min_range, max_range, num_design_points, dim_U)
GP = gp(design_points, vphi(design_points))
    
#Grid points to interpolate with
num_interp_points = num_design_points*4

#%% Calculations
#u lives in [-1,1] so use uniform dist as prior or could use normal with cutoff |x| < 2 
density_prior = uniform_density

flag_run_MCMC = 1
if flag_run_MCMC:
    x0 = np.zeros(dim_U)
    print('Parameter is:', u_dagger)
    print('Solution to PDE at',x,'for true parameter is:', G_u_dagger)
    print('Mean of y for', num_obs,'observations is:', np.sum(y)/(num_obs*dim_U))
    
    if 0:
        density_post = lambda u: np.exp(-phi(u))*density_prior(u)
        name = 'True posterior'
        run_true = runMCMC(density_post, length, speed_random_walk, x0, x, N, name)
    
    if 0:
        density_post = lambda u: np.exp(-GP.mean(u))*density_prior(u)
        name = 'GP as mean - ie marginal approximation'
        run_mean = runMCMC(density_post, length*10, speed_random_walk, x0, x, N, name)
        
    if 1:
        density_post = lambda u: np.exp(-GP.GP_eval(u))*density_prior(u)
        name = 'GP - one evaluation'
        run_GP = runMCMC(density_post, length*10, speed_random_walk, x0, x, N, name)
        plot_dist(run_GP, name)
    
    if 0:
        interp = GP.GP(num_interp_points)
        density_post = lambda u: np.exp(-interp(u))*density_prior(u)
        name = 'pi^N_rand via interpolation'
        run_rand = runMCMC(density_post, length*10, speed_random_walk, x0, x, N, name)

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