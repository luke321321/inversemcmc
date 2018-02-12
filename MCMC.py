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
where u \in [-1,1]^d and the truth u^* is randomly generated.

"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#kv is modified Bessel function of second kind
from scipy.sparse import diags
#from scipy.sparse.linalg import spsolve
#Quicker sparse LA solver: install via: 'conda install -c haasad pypardiso'
from pypardiso import spsolve
#Progress bar
from tqdm import tqdm

#import timeit
#import cProfile
#import pstats

from GaussianProcess import GaussianProcess as gp

#We pass dimension via x0
def MH_random_walk(density, length, speed=0.5, x0=np.array([0]), burn_time=1000):
    #Calculate dim of parameter space
    if isinstance(x0, np.ndarray):
        dim = x0.size
    else:
        dim = 1
        
    x = np.zeros((burn_time + length, dim))
    #Generates normal rv in R^n of length=speed or 0.
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


#%% PDE solver code
def solve_PDE(u,N):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points"""
    
    c = 100.
    
    #create N Chebyshev nodes in (0,1)
    nodes = np.zeros(N+2)
    nodes[1:-1] = np.cos((2*np.arange(N)+1)*np.pi/(2*N) - np.pi)/2 + 0.5
    nodes[-1] = 1
    
    A, b_bttm = stiffness_matrix(nodes,u)
    b = cal_B(nodes)
    
    #change last entry of b for Dirichlet bdry condition at 1
    b[-1] = b[-1] + c*b_bttm
    b = np.append(b, c)
    
    #solve the PDE
    p = spsolve(A, b)
    
    #add 0 for the first node
    p = np.insert(p,0,0)
    return p, nodes

def stiffness_matrix(nodes, u):
    """Returns the sparse stiffness matrix from nodes and k(x;u), nodes include enpoints"""
    
    #calculate derivative of basis functions - which is just a step function
    v_L = 1/(nodes[1:] - nodes[:-1])
    v_R = 1/(nodes[1:] - nodes[:-1])
    
    #integrate K between the nodes
    intK = integral_K(nodes[:-1], nodes[1:], u)
    
    #Construct the stiffness matrix
    diag_0 = (v_L[:-1] ** 2) * intK[:-1] + (v_R[1:] ** 2) * intK[1:]
    diag_1 = -v_R[1:-1] * v_L[1:-1] * intK[1:-1]
    
    #Append values for Dirchlet condition of p(1) = 1
    diag_0 = np.append(diag_0, 1)
    diag_1 = np.append(diag_1, 0)
    
    #get the value bottom of off diagonal to take from vector b
    b_bttm = v_R[-1] * v_L[-1] * intK[-1]
    
    A = diags([diag_1, diag_0, diag_1], [-1, 0, 1], format="csr")
    return A, b_bttm

def cal_B(nodes):
    """Returns the vector b to solve the PDE
    
    If want RHS of PDE to be f(x) then change return to
    f(node[1:-1]) * (nodes[2:] - nodes[:-2])/2 to approximate \int fv using midpoint rule"""
    
    return (nodes[2:] - nodes[:-2])/2

def integral_K(a,b,u):
    """Returns the integral of k(x;u) from a to b, both 1d arrays of the same dimension"""
    if isinstance(u, np.ndarray):
        dim_U = u.size
    else:
        dim_U = 1
    
    #tile arrays to 2d arrays
    A = np.broadcast_to(a, (dim_U, a.size))
    B = np.broadcast_to(b, (dim_U, b.size))
    J = np.broadcast_to(np.arange(dim_U)+1, (a.size, dim_U)).T
    U = np.broadcast_to(u, (a.size, dim_U)).T
    
    #calculate k(x;u) at nodes x
    to_sum = U*(np.cos(2*np.pi*J*A) - np.cos(2*np.pi*J*B))/(2*np.pi*J)
    return (b-a)/100 + np.sum(to_sum, axis=0)/(200*(dim_U + 1))

def solve_PDE_at_x(u,N,x):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points, and returns the value of p(x) where 0 < x < 1"""
    p, nodes = solve_PDE(u,N)
    i = np.searchsorted(nodes, x)
    return (p[i-1]*(x - nodes[i-1])+ p[i]*(nodes[i] - x))/(nodes[i] - nodes[i-1])


def runMCMC(dens, length, speed_random_walk, x0, x, N):
        print('Running MCMC with length:', length, 'and speed:', speed_random_walk)
        accepts, run = MH_random_walk(dens, length, x0=x0, speed=speed_random_walk)
        #cProfile.runctx('MH_random_walk(density_prior, length, x0=x0, speed=speed_random_walk)'
        #                , globals(), locals(), '.prof')
        #s = pstats.Stats('.prof')
        #s.strip_dirs().sort_stats('time').print_stats(30)
        mean = np.sum(run, 0)/length
        print('Mean is:', mean)
        print('We accepted this number of times:', accepts)
        sol_at_mean = solve_PDE_at_x(mean,N,x)
        print('Solution to PDE at mean is:', sol_at_mean)
        return accepts, run

#%% Setup variables and functions
#First neeed to generate some data y
#setup
        
#sigma is noise in observations
sigma = 0.05
dim_U = 1
length = 10**4 #length of random walk in MCMC
num_design_points = 20 #in each dimension
speed_random_walk = 0.1
#End points of n-dim lattice for the design points
min_range = -1
max_range = 1
num_obs = 50

#N: number basis functions for solving PDE
N = 2 ** 15
#point to solve PDE at
x = 0.15

#Generate data
#The truth u_dagger lives in [-1,0.5]
u_dagger = np.random.rand(dim_U) - 1
#u_dagger = -0.7
G_u_dagger = solve_PDE_at_x(u_dagger, N, x)
#y = G_u_dagger
y = np.broadcast_to(G_u_dagger, (num_obs, dim_U)) + sigma*np.random.standard_normal((num_obs, dim_U))

_ROOT2PI = math.sqrt(2*math.pi)
normal_density = lambda x: math.exp(-np.sum(x ** 2)/2)/_ROOT2PI
#uniform density for [-1,1]
uniform_density = lambda x: 1*(np.sum(x ** 2) <= 2)

phi = lambda u: np.sum((y - solve_PDE_at_x(u,N,x)) ** 2)/(2*sigma*num_obs)
vphi = np.vectorize(phi, signature='(i)->()')

#Create Gaussian Process with exp kernel
design_points = gp.create_uniform_grid(min_range, max_range, num_design_points, dim_U)
GP = gp(design_points, vphi(design_points))
    
#Grid points to interpolate with
num_interp_points = num_design_points*4

#%% Calculations
#u lives in [-1,1] so use uniform dist as prior or could use normal with cutoff |x| < 2 
density_prior = uniform_density
density_post = lambda u: np.exp(-GP.mean(u))*uniform_density(u)


flag_run_MCMC = 1
if flag_run_MCMC:
    x0 = np.zeros(dim_U)
    print('Parameter is:', u_dagger)
    print('Solution to PDE at',x,'for true parameter is:', G_u_dagger)
    print('Mean of y for', num_obs,'observations is:', np.sum(y, 0)/num_obs)
    
    if 0:
        density_post = lambda u: np.exp(-phi(u))*density_prior(u)
        print('\n True posterior')
        _, run_true = runMCMC(density_post, length, speed_random_walk, x0, x, N)
    
    if 0:
        density_post = lambda u: np.exp(-GP.mean(u))*density_prior(u)
        print('\n GP as mean - ie marginal approximation')
        _, run_mean = runMCMC(density_post, length*10, speed_random_walk, x0, x, N)
        
    if 1:
        density_post = lambda u: np.exp(-GP.GP_eval(u))*density_prior(u)
        print('\n GP - one evaluation')
        _, run_GP = runMCMC(density_post, length*10, speed_random_walk, x0, x, N)
    
    if 0:
        interp = GP.GP(num_interp_points)
        density_post = lambda u: np.exp(-interp(u))*density_prior(u)
        print('\n pi^N_rand via interpolation')
        _, run_rand = runMCMC(density_post, length*10, speed_random_walk, x0, x, N)


#%% Plotting 
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
        
#%% Plot hist  
flag_plot = 1
if flag_plot:
    plt.figure()
    if dim_U == 1:
        plt.hist(run_GP, bins=101, alpha=0.5, density=True, label='Post')
#        plt.hist(distPost, bins=101, alpha=0.5, density=True, label='Post')
    else:
        plt.hist(run_GP[:,0], bins=101, alpha=0.5, density=True, label='Post')
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

#likelihood for u_dagger in 1d
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
    X = np.linspace(-1,1,40)
    vPDE_at_x = np.vectorize(lambda u: solve_PDE_at_x(u, N, x))
    plt.plot(X,vPDE_at_x(X))
    plt.title('Solution to PDE for different parameters')
    plt.show()

    
#%%Testing code for PDE solver
#u = np.random.randn(1)
#N = 100
#p, nodes = solve_PDE(u,N)
#plt.figure()
#plt.plot(nodes,p)
#plt.show()
#print('Value at 0.5:', solve_PDE_at_x(u,N,0.5))
#cProfile.runctx('solve_PDE_at_x(u,N,x)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)