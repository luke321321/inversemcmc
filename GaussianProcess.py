"""
Created on Thu Jan 18 11:30:39 2018

@author: s1002685
"""

import numpy as np
import math
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt

class GaussianProcess:
    def __init__(self, design_points, data, nu=1/2, sig2 = 1, lam = 1):
        """Initialises a Guassian Emulator with the kernel being a Matern Kernal.
        Defaults: to exponential kernal
        Gaussian: nu = np.inf
        Exp: nu = 1/2
        """

        #if the design_points are 1d append empty axis for r2_distance method
        if len(design_points.shape) == 1:
            self.design_points = design_points[:,np.newaxis]
        else:
            self.design_points = design_points

        self.data = data
        self.nu = nu
        self.sig2 = sig2
        self.lam = lam
        self.mean, self.kernal = self.create_matern()

    def create_matern(self):
        """Creates a Guassian Emulator with the kernel being a Matern Kernal.
        Returns functions mean(u), kernal_N(u,v)

        Note for functions mean(u), kernal_N(u,v)
        If u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector
        and the same for v
        """
        phiStar = self.data

        def k(self, u, v):
            r2 = self.r2_distance(u,v)

            if self.nu == np.inf:
                return np.exp(-r2)
            elif self.nu == 0.5:
                return self.sig2*np.exp(-np.sqrt(r2)/self.lam)
            else:
                rt2_nu = math.sqrt(2*self.nu)/self.lam
                const = (self.sig2/gamma(self.nu)*math.pow(2,self.nu-1))
                kuv = const*np.power(rt2_nu*r2,self.nu)*kv(self.nu,rt2_nu*r2)

                #if r2[i,j] = 0 then want kernalStar[i,j] = 1
                #Asymptotics when r2 = 0 not the best
                where_NaNs = np.isnan(kuv)
                kuv[where_NaNs] = 1
                return kuv

        kernel_star_inverse = np.linalg.inv(k(self,self.design_points, self.design_points))
        kernel_star_inverse_dot_phi_star = kernel_star_inverse @ phiStar

        mean = lambda u : k(self,u, self.design_points) @ kernel_star_inverse_dot_phi_star
        kernal_N = lambda u,v: (k(self,u,v) - k(self,v, self.design_points) @ kernel_star_inverse
                                @ k(self, u, self.design_points).T)

        return mean, kernal_N

    def GP_at_points(self, grid_points, num_evaluations=1):
        """Returns a Gaussian Process evalued at the grid points"""
        return multivariate_normal(self.mean(grid_points), self.kernel(grid_points,grid_points),
                                    allow_singular=True, size=num_evaluations).T

    def GP(self, num_grid_points, interp_method='linear', num_evaluations=1):
        if interp_method == 'linear':
            return self.GP_interp_linear(num_grid_points, num_evaluations)

    def GP_interp_linear(self, num_grid_points, num_evaluations):
        """Returns a function that is a GP evaluated exactly at the grid points
        and used linear intepolation to evaluate it at other points"""

    def Brownian_bridge(self, points, data, cor):
        """Uses a Brownian bridge to interpolate a function at x given
        value at two points x_0 and x_1.
        
        Points: np array (x0, x, x1)
        Data: np array (f(x0), f(x1))"""
        
        T = np.sqrt(self.r2_distance(points[0], points[2]))
        t = np.sqrt(self.r2_distance(points[0], points[1]))
        return data[0] + np.random.normal(scale=cor) + t/T *(data[2]-data[0])


    def create_uniform_grid(self, min_range, max_range, n, dim):
        if dim > 1:
            grid_points_grid = np.meshgrid(*[np.linspace(min_range,max_range,n) for _ in range(dim)])
            grid_points = np.hstack(grid_points_grid).swapaxes(0,1).reshape(dim,-1).T
        else:
            grid_points = np.linspace(min_range, max_range, n)[:,np.newaxis]
        return grid_points

    def r2_distance(self, u, v):
        """Calculates the l^2 distance squared between u and v for u,v being points,
        vectors or list of vectors and returns a point, vector or matrix as appropriate.
        Dimensions of the vectors has to be the same

        Note if u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector.

        If u.shape = (m x d), v.shape = (n x d) then r2.shape = (m x n)
        If u.shape = (m,)     v.shape = (n x d) then r2.shape = (m x n)
        If u.shape = (m x d), v.shape = (n,)    then r2.shape = (m x n)
        If u.shape = (m,)     v.shape = (n,)    then r2.shape = (m x n)
        If u.shape = (d,)     v.shape = (d,)    then r2       = float
        If m or n = 1 then that dimension is squeezed out of r2.shape"""

        #First check dimensions
        dim_U = len(u.shape)
        dim_V = len(v.shape)
        assert dim_U <=2 and dim_V <=2
        
        #first bottom 3 cases from doc
        
        #deal with floats sensibly
        if dim_U == 0 or dim_V == 0:
            diff = u-v
            r2 = np.square(diff)
        elif (dim_U == 1 and dim_V == 1):
            #if 4th case append axes to get correct shape
            if u.shape[0] != v.shape[0]:
                V = v[np.newaxis,:]
                U = u[:,np.newaxis]
                r2 = np.squeeze(np.square(U-V))
            else:
                r2 = np.sum(np.square(u-v))
        else:
            #Always put a new axes at start of v and middle of u
            V = v[np.newaxis,...]
            if dim_U == 1:
                U = u[:, np.newaxis, np.newaxis]
            else:
                U = u[:, np.newaxis, :]
            if dim_V == 1:
                V = V[...,np.newaxis]
            diff = U-V
            r2 = np.squeeze(np.einsum('ijk,ijk->ij',diff,diff))
        return r2


#%% Testing Brownian Bridge code
#GP = GaussianProcess(np.array([0,2]),np.array([1,3]))
#points = np.array([0,0.5,1])
#data = np.array([0,0,1.])
#plt.figure()
#for i in range(30):
#    data[1] = GP.Brownian_bridge(points,data,0.5)
#    plt.plot(points, data)
#plt.show()