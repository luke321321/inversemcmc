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
from scipy.stats import lognorm

from GaussianProcess import GaussianProcess as gp
import PDE_CGSSZ as PDE
from MCMC import runMCMC

def MCMC_helper(density_post, name, shot_name):
    return runMCMC(density_post, length, speed_random_walk, x0, x, N, name, short_name,
                   PDE, G_k_dagger, y, burn=burn_time)

def save_data(run, shot_name):
    mean = np.sum(run, 0)/length
    sol_at_mean = PDE.solve_at_x(mean, N, x)
    np.savez_compressed('output_CGSSZ_' + short_name, run=run, k_dagger=k_dagger,
                        G_k_dagger=G_k_dagger, y=y, sol_at_mean=sol_at_mean, dim_k=dim_k,
                        length=length, sigma=sigma, burn_time=burn_time,
                        speed_random_walk=speed_random_walk, num_obs=num_obs, N=N,
                        num_design_points=num_design_points)
    
#%% Setup variables and functions
sigma = np.sqrt(10 ** -1) #size of the noise in observations
dim_k = 3
length = 10 ** 5 #length of MCMC
burn_time = 3000
num_design_points = 20 #in each dimension
speed_random_walk = 0.2
num_obs = 10
#num_obs evenly spaced points in (0,1)
x = np.arange(1, num_obs + 1)/(num_obs + 1)
#N: number basis functions for solving PDE
N = 2 ** 10

#Generate data
#The truth k_dagger
k_dagger = np.random.lognormal(size=dim_k)
#sol u for k_dagger at points x
G_k_dagger = PDE.solve_at_x(k_dagger, N, x)
y = G_k_dagger + sigma*np.random.normal(size=num_obs)

phi = lambda k: np.sum((y - PDE.solve_at_x(k, N, x)) ** 2)/(2*(sigma**2))
vphi = np.vectorize(phi, signature='(i)->()')

#Have the design_points so they are log normally distributed ie do inverse of cdf
#can't have all 0's being a design point otherwise PDE isn't solvable
design_points = gp.create_uniform_grid(1e-14, 1-1/num_design_points, num_design_points, dim_k)
design_points = lognorm.ppf(design_points, 1)

#Create Gaussian Process with exp kernel
GP = gp(design_points, vphi(design_points))

#%% Calculations
#Note scipy lognormal pdf doesn't seem to check if u > 0 so do this myself
density_prior = lambda u: lognorm.pdf(np.sqrt(np.sum(u**2)), 1)*(np.all(u > 0))

flag_run_MCMC = 1
if flag_run_MCMC:
    #let x0 be the mean of lognormal(0, 1)
    x0 = np.exp(np.ones(dim_k)/2)
    print('Parameter is:', k_dagger)
    print('Solution to PDE at',x,'for true parameter is:')
    print(G_k_dagger)    
    
    if 0:
        density_post = lambda u: np.exp(-phi(u))*density_prior(u)
        name = 'True posterior'
        short_name = 'true'
        run_true = MCMC_helper(density_post, name, short_name)
        save_data(run_true, short_name)
    
    if 0:
        density_post = lambda u: np.exp(-GP.mean(u))*density_prior(u)
        name = 'GP as mean - ie marginal approximation'
        short_name = 'mean'
        run_mean = MCMC_helper(density_post, name, short_name)
        save_data(run_mean, short_name)
        
    if 1:
        density_post = lambda u: np.exp(-GP.GP_eval(u))*density_prior(u)
        name = 'GP - one evaluation'
        short_name = 'GP'
        run_GP = MCMC_helper(density_post, name, short_name)
        save_data(run_GP, short_name)
    
    if 0:
        #Grid points to interpolate with
        num_interp_points = 4 * num_design_points
        interp = GP.GP(num_interp_points)
        density_post = lambda u: np.exp(-interp(u))*density_prior(u)
        name = 'pi^N_rand via interpolation'
        short_name = 'interp'
        run_rand = MCMC_helper(density_post, name, short_name)
        save_data(run_rand, short_name)