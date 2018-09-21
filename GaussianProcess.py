import numpy as np
import math
import numbers
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
from scipy.interpolate import RegularGridInterpolator, interp1d

class GaussianProcess:
    def __init__(self, points, data, total_calls = 1000, nu=1/2, sig2 = 1, lam = 1):
        """Initialises a Guassian Emulator with the kernel being a Matern kernel.
        Defaults: to exponential kernel
        Gaussian: nu = np.inf
        Exp: nu = 1/2
        """
        #tolerance for checking if something is close to 0
        self.tol = 1e-14

        self.points, self.data = self.reshape_points(points, data)
        
        self.data_length = len(data)
        
        self.nu = nu
        self.sig2 = sig2
        self.lam = lam
        self.dim = self.points.shape[1]
        self.total_calls = max(total_calls, self.points.shape[0])
        
        self.mean, self.kernel = self.create_matern()
         
        self.reset()
        
    @staticmethod
    def reshape_points(points, data):
        """if the points are 1d: sort them and append empty axis for r2_distance method
        Returns points and data after shaping and sorting"""
        if len(points.shape) == 1:
            index = np.argsort(points)
            return points[index, np.newaxis], data[index]
        else:
            return points, data
            
        
    def create_matern(self, data=None, points=None):
        """Creates a Guassian Emulator with the kernel being a Matern kernel.
        Returns functions mean(u), kernel_N(u,v)

        Note for functions mean(u), kernel_N(u,v)
        If u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector
        and the same for v
        """
        
        if data is None:
            data = self.data
        if points is None:
            points = self.points
            
        #check points are shaped correctly
        
        if len(points) != 1:
            kernel_star_inverse = np.linalg.inv(self.k(points, points))
            kernel_star_inverse_dot_phi_star = kernel_star_inverse @ data

            mean = lambda u : self.k(u, points) @ kernel_star_inverse_dot_phi_star
            kernel = lambda u,v: self.k(u, v) - (self.k(v, points) @ kernel_star_inverse
                                @ self.k(u, points).T)
        #if only 1 point is given then kernel_star_inverse = 1 and objects are scalars
        else:
            mean = lambda u : self.k(u, points) * data
            kernel = lambda u,v: self.k(u, v) - self.k(v, points) * self.k(u, points).T
            
        return mean, kernel
    
    def k(self, u, v):
        r2 = GaussianProcess.r2_distance(u,v)

        if self.nu == np.inf:
            return np.exp(-r2)
        elif self.nu == 0.5:
            return self.sig2*np.exp(-np.sqrt(r2)/self.lam)
        else:
            rt2_nu = math.sqrt(2*self.nu)/self.lam
            const = (self.sig2/gamma(self.nu)*math.pow(2, self.nu-1))
            kuv = const*np.power(rt2_nu*r2, self.nu)*kv(self.nu, rt2_nu*r2)

            #if r2[i,j] = 0 then want kernelStar[i,j] = 1
            #Asymptotics when r2 = 0 not the best
            where_NaNs = np.isnan(kuv)
            kuv[where_NaNs] = 1
            return kuv

    def GP_at_points(self, points, num_evals=1):
        """Returns a Gaussian Process evalued at the grid points"""
        mean = self.mean(points)
        kernel = self.kernel(points, points)
        val = np.random.multivariate_normal(mean, kernel, num_evals)
        if num_evals > 1:
            val = val.T
        return val

    def GP_interp(self, num_grid_points, num_evals=1):
        """Returns a function that is a GP evaluated exactly at the grid points
        and uses linear intepolation to evaluate it at other points
        
        The returned function is really a vector of functions of size num_evaluations"""
        #TODO: only currently returns 1 evaluation of the GP need to call again for another eval
        if num_evals != 1:
            raise Exception('Not yet implemented')
        dp_min = np.amin(self.points)
        dp_max = np.amax(self.points)
        grid = self.create_uniform_grid(dp_min, dp_max, num_grid_points, self.dim)
        if self.dim == 1:
            return interp1d(grid.squeeze(), self.GP_at_points(grid).squeeze(), copy=False, assume_sorted=True)
        else:
            #reshape grid and data to fit the interpolator method
            grid_points_grid = [np.linspace(dp_min, dp_max, num_grid_points) for _ in range(self.dim)]
            GP_eval = self.GP_at_points(grid).reshape([num_grid_points for _ in range(self.dim)])
            return RegularGridInterpolator(grid_points_grid, GP_eval)
        

    def reset(self):
        """Setup empty arrays to store data and coordinates in for GP_eval method"""
        if self.dim == 1:
            self.X = np.concatenate((self.points[:,0], 
                                np.zeros(self.total_calls - self.data_length, dtype=self.points.dtype)))
        else:
            self.X = np.concatenate((self.points, 
                                     np.zeros((self.total_calls - self.data_length, self.dim), dtype=self.points.dtype)))
        self.Y = np.concatenate((self.data, 
                                np.zeros(self.total_calls - self.data_length, dtype=self.data.dtype)))
        self.index = self.data_length
        
    def get_data(self):
        """Returns the non-zero data and points"""
        X = self.X[:self.index]
        Y = self.Y[:self.index]

        return X, Y
    
    def plot_1d(self):
        """Plots Gaussian Process (in 1d)"""
        
        #Sort to plot nicer
        X,Y = self.get_data()
        ind = np.argsort(X.flatten())
        X = X.flatten()[ind]
        Y = Y[ind]
        
        #Plot results:
        plt.figure()
        plt.plot(X, Y)
        plt.show()
    
    def GP_eval(self, x, save=True, num_evals=1):
        """
        Evaluates the GP at the point x and saves this point to the list of generated points.
        If save=False and num_evals>1 then returns num_evals of GP at x and doesn't save any of them.
        These options are used for the marginal approximation.
        """
        #first check lengeth of array X and Y and expand if necessary
        self.check_mem()
        
        #make sure shape of x is (1, dim)
        if len(x.shape) == 1:
            x = x.reshape(1, self.dim)
        
        #Haven't removed save=True variable to make developer explicitly aware of this behaviour
        if save is True and num_evals is not 1:
            raise ValueError('When save=True then num_evals must be 1.')
        
        """First find closest 3*d points to x
        Code is good in 1d but bad in n dim
        Ideally create RTree but first do naive thing.
        Runs in O(n) time instead of O(log(n)) for RTree insertion and nearest neighbour
        Really only need smallest convex hull in self.X that contains x"""
        
        ind, points = self.find_closest(x)
        data = self.Y[ind]
        
        #check to see if close enough already
        dist = self.r2_distance(x, points)
        ind_min_dist = np.argmin(dist)
        if len(ind) == 1:
            if dist < self.tol:
                data_new = self.Y[ind]
            else:
                data_new = self.GP_bridge(x, points, data)
        elif dist[ind_min_dist] < self.tol:
            data_new = self.Y[ind_min_dist]
        else:
            data_new = self.GP_bridge(x, points, data, size=num_evals)
            if save is True:
                self.add_data(ind, x, data_new)
        return data_new
      
    def find_closest(self, x):
        """Finds closest 3*dim points to x in each signed direction
        Returns: ind, points
        index an array of lists and the closest points 
        """
        X = self.X[:self.index]
        if self.dim == 1:
            ind = np.searchsorted(X, x, side='left')
            ind = np.squeeze(ind).tolist()
            if ind != 0 and ind != self.index:
                #if not 0 or N
                ind = [ind-1, ind]
            elif ind == 0:
                ind = [ind]
            else:
                ind = [ind-1]
            points = X[ind]
            points = points[:, np.newaxis]
        else:
            """Idea: In each 2*d axis direction find nearest point to x and choose it (by its index).
            If no point in that direction then doesn't matter"""
            dist = self.r2_distance(x, X)
            
            #use set structure to easily ensure values are unique
            index = set()
            #first add closest 2*d points then check to see if missed any in any axis and add extra if needed
            #really add 3*d points to try and make sure we don't have to loop through things
            num_close = min(3*self.dim, self.index)
            ind_close = np.argpartition(dist, num_close)
            for i in ind_close[:num_close]:
                index.add(i)
            
            diff = None
            for d in range(self.dim):
                #check to see if element in index is a candidate for that dim
                ind_pos = None
                ind_neg = None
                for i in index:
                    tmp = (x-X[i])[:, d]
                    if tmp > 0:
                        ind_pos = i
                    elif tmp < 0:
                        ind_neg = i
                #if no suitable candiate found then do it the slow way
                if ind_pos is None or ind_neg is None:
                    if diff is None:
                        diff = x - X
                    if ind_pos is None:
                        ind_pos = np.nonzero(diff[:, d] > 0)
                        if ind_pos[0].size > 0:
                            dist_pos = dist[ind_pos]         
                            closest_ind = np.argmin(dist_pos)
                            index.add((ind_pos[0][closest_ind]))
                    if ind_neg is None:
                        ind_neg = np.nonzero(diff[:, d] < 0)
                        if ind_neg[0].size > 0:
                            dist_neg = dist[ind_neg]          
                            closest_ind = np.argmin(dist_neg)
                            index.add((ind_neg[0][closest_ind]))
            ind = list(index)
            points = X[ind]
        
        return ind, points
    
    def add_data(self, ind, x, data):
        """Adds new point x and corresponding data to the appropriate memory structure"""
        #add new data to class
        if self.dim == 1:
            #make sure index is leftmost
            if len(ind) == 2:
                ind_L = ind[1]
            else:
                ind_L = ind[0]
            
            if len(ind) == 1 and ind[0]+1 == self.index:
                 #if we're at the last element just insert
                 self.X[self.index] = x
                 self.Y[self.index] = data
            else:
                #shift current elements by 1
                self.X[ind_L+1:self.index+1] = self.X[ind_L:self.index]
                self.Y[ind_L+1:self.index+1] = self.Y[ind_L:self.index]
                self.X[ind_L] = x
                self.Y[ind_L] = data
        else:
            self.X[self.index] = x
            self.Y[self.index] = data
            
        self.index += 1
    
    def check_mem(self):
        """Checks if storage arrays are big enough and resizes if needed"""
        if(self.index == len(self.Y)):
            #double length of arrays
            self.X = np.concatenate((self.X, np.zeros(self.X.shape, dtype=self.X.dtype)))
            self.Y = np.concatenate((self.Y, np.zeros(self.Y.shape, dtype=self.Y.dtype)))
    
    def GP_bridge(self, x, points, data, size=1):
        """Uses a small Gaussian process to evaluate the GP at x given it at points
        
        x: point to calculate the GP at
        Points: np array (x0, x1, ...)
        Data: np array (f(x0), f(x1), ...)
        size: the number of evaluations to return"""
        
        mean, ker = self.create_matern(data = data, points = points)

        if len(x.shape) == 1:
            x = x.reshape(1, self.dim)
        all_pts = np.concatenate((x, points))
        m = np.squeeze(mean(all_pts))
        k = ker(all_pts, all_pts)
        val = np.random.multivariate_normal(m, k, size, check_valid='ignore')
        if size == 1:
            return val[0,0]
        else:
            return val[:,0]

    @staticmethod
    def create_uniform_grid(min_range, max_range, n, dim=1):
        """Creates a uniform grid with n points in each dimension on 
        [min_range, max_range]^dim"""
        
        if dim > 1:
            grid_points_grid = np.meshgrid(*[np.linspace(min_range, max_range, n) for _ in range(dim)])
            grid_points = np.hstack(grid_points_grid).swapaxes(0, 1).reshape(dim, -1).T
        else:
            grid_points = np.linspace(min_range, max_range, n)[:, np.newaxis]
        return grid_points
    
    @staticmethod
    def r2_distance(u, v):
        """Calculates the l^2 distance squared between u and v for u,v being points,
        vectors or list of vectors and returns a point, vector or matrix as appropriate.
        Dimensions of the vectors has to be the same

        Note if u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector.

        If u.shape = (m x d), v.shape = (n x d) then r2.shape = (m x n)
        If u.shape = (d,)     v.shape = (n x d) then r2.shape = (n)
        If u.shape = (m x d), v.shape = (d,)    then r2.shape = (m)
        If u.shape = (m,)     v.shape = (n,)    then r2.shape = (m x n)
        If u.shape = (d,)     v.shape = (d,)    then r2       = float
        If m or n = 1 then that dimension is squeezed out of r2.shape"""
                
        #First deal with floats sensibly and return
        if isinstance(u, numbers.Number) or isinstance(v, numbers.Number):
            r2 = np.squeeze(np.square(u-v))
            return r2

        #Now check dimensions and inputs
        dim_U = len(u.shape)
        dim_V = len(v.shape)

        #Can remove these asserts to speed up the function
        assert dim_U <= 2 and dim_V <= 2
        if dim_U == 2 and dim_V == 2:
            assert u.shape[1] == v.shape[1]
        elif dim_U == 1 and dim_V == 2:
            assert u.shape[0] == v.shape[1]
        elif dim_U == 2 and dim_V == 1:
            assert u.shape[1] == v.shape[0]
        
        if dim_U == 1 and dim_V == 1:
            #if 4th case append axes to get correct shape
            if u.shape[0] != v.shape[0]:
                V = v[np.newaxis,:]
                U = u[:,np.newaxis]
                r2 = np.squeeze(np.square(U - V))
            else:
                r2 = np.sum(np.square(u - v))
        elif dim_U < 2 or dim_V < 2:
            #cases 3 and 4, make u have the smallest dimension
            if dim_U > dim_V:
                u, v = v, u
            U = u[np.newaxis, :]
            r2 = np.sum(np.square(U - v), 1)
        else:
            #Deal with large u and v incase of out of memory errors
            try:
                U = u[:, np.newaxis, :]
                V = v[np.newaxis,...]
                r2 = np.squeeze(np.sum(np.square(U - V), 2))
            except MemoryError:
                #try to sum a slower way instead without creating a large intermediate array
                r2 = np.zeros((u.shape[0], v.shape[0]))
                if u.shape[0] < v.shape[0]:
                    for i in range(u.shape[0]):
                        r2[i,:] = np.sum(np.square(u[i][np.newaxis, :] - v), 1)
                else:
                    for j in range(v.shape[0]):
                        r2[:,j] = np.sum(np.square(u - v[j][np.newaxis, :]), 1)
                r2 = np.squeeze(r2)
        return r2