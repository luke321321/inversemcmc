"""
PDE solving code for solving following PDE in p:
-d/dx (k(x) du/dx(x; k)) = 4x in (0,1)
u(0; k) = 0
u(1; k) = 1

log(k(x))  piecewise constant over 10 equally spaced intervals
with k_0 = 1
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
#Quicker sparse LA solver: install via: 'conda install -c haasad pypardiso'
#from pypardiso import spsolve

def solve(k, N):
    """Solves the PDE in (0,1) with coefficients k and
    N number of Chebyshev interpolant points"""
    
    #Make sure k_0 = 1 (or however many dimesions we are not searching for)
    k_full = np.concatenate((np.ones(10-k.shape[0]), k))
    
    #Boundary condition, p(1; u) = c
    c = 1.
    
    #create N Chebyshev nodes in (0,1)
    nodes = np.zeros(N+2)
    nodes[1:-1] = np.cos((2*np.arange(N)+1)*np.pi/(2*N) - np.pi)/2 + 0.5
    nodes[-1] = 1
    
    A, b_bttm = stiffness_matrix(nodes, k_full)
    b = cal_B(nodes)
    
    #change last entry of b for Dirichlet bdry condition at 1
    b[-1] = b[-1] + c*b_bttm
    b = np.append(b, c)
    
    #solve the PDE
    p = spsolve(A, b)
    
    #add 0 for the first node
    p = np.insert(p,0,0)
    return p, nodes

def stiffness_matrix(nodes, k):
    """Returns the sparse stiffness matrix from nodes and k(x), nodes include enpoints"""
    
    #calculate derivative of basis functions - which is just a step function
    v_L = 1/(nodes[1:] - nodes[:-1])
    v_R = v_L
    
    #integrate K between the nodes
    intK = integral_K(nodes[:-1], nodes[1:], k)
    
    #Construct the stiffness matrix
    diag_0 = (v_L[:-1] ** 2) * intK[:-1] + (v_R[1:] ** 2) * intK[1:]
    diag_1 = -v_R[1:-1] * v_L[1:-1] * intK[1:-1]
    
    #Append values for Dirchlet condition of u(1) = c
    diag_0 = np.append(diag_0, 1)
    diag_1 = np.append(diag_1, 0)
    
    #get the value bottom of off diagonal to take from vector b
    b_bttm = v_R[-1] * v_L[-1] * intK[-1]
    
    A = diags([diag_1, diag_0, diag_1], [-1, 0, 1], format="csr")
    return A, b_bttm

def cal_B(nodes):
    """Returns the vector b to solve the PDE
    
    If want RHS of PDE to be f(x) then change return to
    f(nodes[1:-1]) * (nodes[2:] - nodes[:-2])/2 to approximate \int fv using midpoint rule"""
    
    return 4*(nodes[1:-1]) * (nodes[2:] - nodes[:-2])/2

def integral_K(a, b, k):
    """Returns the integral of k(x;u) from a to b, both 1d arrays of the same dimension
    Does this via approximation with the midpoint rule"""
    dim_K = k.size
    midpoint = a + (b-a)/2
    ind_k = (midpoint*dim_K).astype(int)
    
    return (b-a) * k[ind_k]

def solve_at_x(k, N, x):
    """Solves the PDE in (0,1) with coefficients k and
    N number of Chebyshev interpolant points, and returns the value of u(x) where 0 < x < 1"""
    p, nodes = solve(k, N)
    i = np.searchsorted(nodes, x)
    return (p[i-1]*(x - nodes[i-1])+ p[i]*(nodes[i] - x))/(nodes[i] - nodes[i-1])

#%%Testing code for PDE solver
#k = np.random.lognormal(size=4)
#N = 10000
#u, nodes = solve(k, N)
#plt.figure()
#plt.plot(nodes, u)
#plt.show()
#print('Value at 0.5:', solve_at_x(k, N, 0.5))
#cProfile.runctx('solve_at_x(u,N,0.25)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)