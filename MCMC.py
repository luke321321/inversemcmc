"""
Code for running MCMC
"""

import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt

#Progress bar
from tqdm import tqdm

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
    for i in tqdm(range(1, burn_time + length)):
        y = x[i-1] + rvNormal[i]
        if(density_old == 0): #Accept whatever
            x[i] = y
            density_old = density(y)
            if i > burn_time:
                accepted_count += 1
        else:
            u = random.random()
            density_new = density(y)
            if(u <= min(1, density_new/density_old)): #check acceptance
                x[i] = y
                density_old = density_new
                if i > burn_time:
                    accepted_count += 1
            else:
                x[i] = x[i-1]
    return accepted_count, x[burn_time:]

def runMCMC(dens, length, speed_random_walk, x0, x, N, name, PDE, burn=1000):
    """Helper function to start off running MCMC"""
    print('\n' + name)
    print('Running MCMC with length:', length, 'and speed:', speed_random_walk)
    accepts, run = MH_random_walk(dens, length, x0=x0, speed=speed_random_walk, burn_time=burn)

    mean = np.sum(run, 0)/length
    print('Mean is:', mean)
    print('We accepted this number of times:', accepts)
    sol_at_mean = PDE.solve_at_x(mean, N, x)
    print('Solution to PDE at mean is:', sol_at_mean)
    plot_dist(run, name)
    np.save("MCMC_run.npy", run)
    return run

def plot_dist(dist, title):
    """Plots the distribution on a grid"""
    sns.set(color_codes=True)
    sns.set_style('ticks')
    g = sns.PairGrid(pd.DataFrame(dist), despine=True)
    g.map_diag(sns.kdeplot, legend=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d", n_levels=5)
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    g.savefig("output.png")