"""PDE Solving code for solving following PDE in p:
-d/dx (k(x; u) dp/dx(x; u)) = 1 in (0,1)
p(0; u) = 0
p(1; u) = 30

k(x; u) = 1/100 + \sum_j^d u_j/(200(d + 1)) * sin(2\pi jx)"""


import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
#Quicker sparse LA solver: install via: 'conda install -c haasad pypardiso'
#from pypardiso import spsolve

import matplotlib.pyplot as plt

def solve(u,N):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points"""
    
    #Boundary condition, p(1; u) = c
    c = 30.
    
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

def solve_at_x(u,N,x):
    """Solves the PDE in (0,1) with coefficients u and
    N number of Chebyshev interpolant points, and returns the value of p(x) where 0 < x < 1"""
    p, nodes = solve(u,N)
    i = np.searchsorted(nodes, x)
    return (p[i-1]*(x - nodes[i-1])+ p[i]*(nodes[i] - x))/(nodes[i] - nodes[i-1])

#%%Testing code for PDE solver
#u = np.random.randn(size=3)
#N = 100
#p, nodes = solve(u,N)
#plt.figure()
#plt.plot(nodes,p)
#plt.show()
#print('Value at 0.5:', solve_at_x(u,N,0.5))
#cProfile.runctx('solve_at_x(u,N,x)'
#                , globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)