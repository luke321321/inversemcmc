"""
Created on Thu Jan 18 11:30:39 2018

@author: s1002685
"""

import numpy as np
import math
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator, interp1d

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
        
        kernel_star_inverse = np.linalg.inv(k(self.design_points, self.design_points, self.nu))
        kernel_star_inverse_dot_phi_star = kernel_star_inverse @ phiStar
        
        mean = lambda u : k(u, self.design_points) @ kernel_star_inverse_dot_phi_star
        kernal_N = lambda u,v: (k(u,v) - k(v, self.design_points) @ kernel_star_inverse 
                                @ k(u, self.design_points).T)
    
        return mean, kernal_N
    
    def GP_on_grid(self, grid_points, num_evaluations=1):
        """Returns a Gaussian Process evalued at the points"""
        return multivariate_normal(self.mean(grid_points), self.kernel(grid_points,grid_points),
                                    allow_singular=True, size=num_evaluations).T
                                   
    def GP(self, num_grid_points, interp_method='linear', num_evaluations=1):
        if interp_method == 'linear':
            return self.GP_interp_linear(num_grid_points, num_evaluations)
        
    def GP_interp_linear(self, num_grid_points, num_evaluations):
        """Returns a function that is a GP evaluated exactly at the grid points
        and used linear intepolation to evaluate it at other points"""
        
        
        
    
    
    def create_uniform_grid(min_range, max_range, n, dim):
        if dim > 1:
            grid_points_grid = np.meshgrid(*[np.linspace(min_range,max_range,n) for _ in range(dim)])
            grid_points = np.hstack(grid_points_grid).swapaxes(0,1).reshape(dim,-1).T
        else:
            grid_points = np.linspace(min_range, max_range, n)[:,np.newaxis]
        return grid_points        
    
    def r2_distance(u, v):
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
        
    